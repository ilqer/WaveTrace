// Unified mesh node = STA (router backhaul + channel lock) + ESP-NOW (sensing traffic).
//
// FULL-MESH via TIME-DIVISION ROUND-ROBIN. A single 2.4 GHz radio is half-duplex, so true
// simultaneous TX+RX is impossible. Instead exactly ONE node transmits per turn (an ESP-NOW
// broadcast burst) while every other node captures CSI from it; the TX role rotates through all
// LIVE nodes. N nodes => N*(N-1) directed (tx,rx) links per cycle.
//
// DYNAMIC RING: there is no compile-time node count. Each node learns its peers from their frames
// (with a liveness timeout); the ring is the sorted set of currently-alive ids, the leader is the
// lowest live id, and the token is handed to the next-higher live id (wrapping). Boards can be
// added/removed live — 2 -> 1>2>1, 3 -> 1>2>3>1, etc., with no reflash.
//
// Turn-taking = TOKEN PASSING (the burst's last frame names the next id). The current leader
// bootstraps and self-heals a dropped token. SNTP stamps CSI with a shared wall clock so the PC can
// align frames across nodes (no internet: PC runs ntp_server.py) — it never drives the rotation.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "esp_wifi.h"
#include "esp_now.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "esp_netif_sntp.h"
#include "esp_sntp.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "lwip/sockets.h"
#include "mdns.h"
#include "driver/gpio.h"
#include "esp_rom_gpio.h"
#include "esp_csi_gain_ctrl.h"
#include "config.h"
#if STATUS_LED_GPIO >= 0
#include "led_strip.h"
#endif

#ifndef NODE_ID
#define NODE_ID 1
#endif

#define MESH_MAGIC 0x57            // 'W' — marks our mesh frames, ignore foreign ESP-NOW
#define FLAG_LAST  0x01            // set on the final frame of a burst (carries the token handoff)
#define UDP_BATCH_BYTES 8192       // max CSI body per datagram before a forced flush

static const char *TAG = "wt_node";
static const uint8_t BCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

// Token frame: 5 bytes, broadcast. next_id on the LAST frame hands the turn to that node.
typedef struct __attribute__((packed)) {
    uint8_t magic;
    uint8_t tx_id;
    uint8_t seq;
    uint8_t flags;
    uint8_t next_id;
} mesh_pkt_t;

static char data_buf[384 * 8 + 8];
static QueueHandle_t s_csi_q;        // char* CSI_DATA lines -> udp_batch_task
static QueueHandle_t s_turn_q;       // "go" tokens -> mesh_task
static volatile uint32_t s_csi_count = 0;
static volatile uint32_t s_csi_raw = 0;   // DIAG: every CSI callback, before the MAC filter
static volatile uint32_t s_tx_sent = 0;   // DIAG: ESP-NOW frames we successfully queued for TX
static volatile uint32_t s_last_air_ms = 0;   // last time any mesh frame was heard
static volatile bool s_connected = false;     // STA associated + has IP (gate mesh TX on this)
static struct in_addr s_pc_addr = {0};        // discovered PC address

static inline uint32_t now_ms(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

// PC discovery: listen for "WAVETRACE_PING" broadcast from health_monitor.py to find its IP.
static void discovery_task(void *) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in serv = {};
    serv.sin_family = AF_INET;
    serv.sin_port = htons(DISCOVERY_PORT);
    serv.sin_addr.s_addr = htonl(INADDR_ANY);
    bind(sock, (struct sockaddr *)&serv, sizeof(serv));
    char buf[32];
    inet_pton(AF_INET, PC_IP, &s_pc_addr); // default fallback
    for (;;) {
        struct sockaddr_in from;
        socklen_t len = sizeof(from);
        int n = recvfrom(sock, buf, sizeof(buf) - 1, 0, (struct sockaddr *)&from, &len);
        if (n > 0) {
            buf[n] = '\0';
            if (strstr(buf, "WAVETRACE_PING")) {
                if (from.sin_addr.s_addr != s_pc_addr.s_addr) {
                    s_pc_addr = from.sin_addr;
                    ESP_LOGI(TAG, "discovered pc at %s", inet_ntoa(s_pc_addr));
                    esp_sntp_stop();
                    esp_sntp_setservername(0, inet_ntoa(s_pc_addr));
                    esp_sntp_init();
                }
            }
        }
    }
}

// PHY gain lock (decided PRIMARY, plan §5 Phase-0): collect a quiet baseline of AGC/FFT gain, then
// force it in-chip so CSI amplitude is comparable across packets/sessions. Host GainLock = fallback.
#define GAIN_BASELINE_PKTS 300
static volatile uint32_t s_gain_samples = 0;
static volatile int s_gain_locked = 0;        // DIAG: 0=collecting baseline, 1=gain forced, 2=skipped
static volatile uint8_t s_lock_agc = 0;       // DIAG: the locked AGC/FFT values
static volatile int8_t s_lock_fft = 0;

// MAC -> node id + last-heard time, learned from ESP-NOW frames. csi_cb only knows the sender MAC,
// so this resolves which node TRANSMITTED a captured frame AND who is currently alive (the ring).
static uint8_t s_macs[MAX_NODES][6];
static uint8_t s_ids[MAX_NODES];
static volatile uint32_t s_seen_ms[MAX_NODES];
static volatile int s_nmac = 0;

static void learn_mac(const uint8_t *mac, uint8_t id) {
    uint32_t t = now_ms();
    for (int i = 0; i < s_nmac; i++) if (s_ids[i] == id) { s_seen_ms[i] = t; return; }
    if (s_nmac >= MAX_NODES) return;
    memcpy(s_macs[s_nmac], mac, 6);
    s_ids[s_nmac] = id;
    s_seen_ms[s_nmac] = t;
    s_nmac++;
}

static int id_for_mac(const uint8_t *mac) {
    for (int i = 0; i < s_nmac; i++) if (memcmp(s_macs[i], mac, 6) == 0) return s_ids[i];
    return -1;  // not a mesh node -> not our sensing traffic
}

// --- Dynamic ring helpers (self is always alive; peers expire after LIVE_TIMEOUT_MS) ---
static int alive_peers(void) {
    uint32_t t = now_ms(); int c = 0;
    for (int i = 0; i < s_nmac; i++)
        if (s_ids[i] != NODE_ID && t - s_seen_ms[i] < LIVE_TIMEOUT_MS) c++;
    return c;
}
// Lowest live id (incl self) = the leader.
static int leader_id(void) {
    uint32_t t = now_ms(); int best = NODE_ID;
    for (int i = 0; i < s_nmac; i++)
        if (t - s_seen_ms[i] < LIVE_TIMEOUT_MS && s_ids[i] < best) best = s_ids[i];
    return best;
}
// Next live id strictly greater than `me`, else wrap to the lowest live id. Self is a candidate.
static int next_after(int me) {
    uint32_t t = now_ms();
    int best_gt = 0x7fffffff, best_any = NODE_ID;
    for (int i = 0; i < s_nmac; i++) {
        if (t - s_seen_ms[i] >= LIVE_TIMEOUT_MS) continue;
        int id = s_ids[i];
        if (id > me && id < best_gt) best_gt = id;
        if (id < best_any) best_any = id;
    }
    if (NODE_ID > me && NODE_ID < best_gt) best_gt = NODE_ID;  // self
    return best_gt != 0x7fffffff ? best_gt : best_any;
}

// Shared epoch-ms once SNTP has synced; monotonic esp_timer ms as a fallback before then.
static uint64_t wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    if (tv.tv_sec > 1600000000)  // plausible real time -> SNTP synced
        return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
    return (uint64_t)(esp_timer_get_time() / 1000);
}

// Capture CSI; emit one esp-csi CSV line (sender MAC = tx identity) to the batch task. No printf/
// sendto here (runs in the Wi-Fi task — blocking it starves the WDT).
static void csi_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) return;
    s_csi_raw++;  // DIAG: count every CSI callback regardless of source
    int tx = id_for_mac(info->mac);
    if (tx < 0) return;  // only emit CSI from known mesh nodes
    int8_t *buf = info->buf;
    int S = info->len / 2;
    if (S <= 0 || S > 384) return;
#if WIFI_CSI_PHY_GAIN_ENABLE
    // Freeze AGC/FFT once a baseline is built (room must stay still for the first few seconds).
    if (!s_gain_locked) {
        uint8_t agc; int8_t fft;
        esp_csi_gain_ctrl_get_rx_gain(&info->rx_ctrl, &agc, &fft);
        if (s_gain_samples < GAIN_BASELINE_PKTS) {
            esp_csi_gain_ctrl_record_rx_gain(agc, fft);
            s_gain_samples++;
        } else {
            uint8_t bagc; int8_t bfft;
            if (esp_csi_gain_ctrl_get_rx_gain_baseline(&bagc, &bfft) == ESP_OK) {
                s_lock_agc = bagc; s_lock_fft = bfft;
                // ESPectre: AGC < 30 = signal too strong; forcing it can make the driver fail to
                // decode (CSI collapse / WDT). Skip the lock and rely on host CV normalization.
                if (bagc < 30) {
                    s_gain_locked = 2;  // SKIP (too strong)
                } else {
                    esp_csi_gain_ctrl_set_rx_force_gain(bagc, bfft);
                    s_gain_locked = 1;  // LOCK
                }
            } else {
                s_gain_samples = 0;  // baseline not ready yet -> recollect
            }
        }
    }
#endif
    char mac[18];
    snprintf(mac, sizeof(mac), "%02x:%02x:%02x:%02x:%02x:%02x",
             info->mac[0], info->mac[1], info->mac[2], info->mac[3], info->mac[4], info->mac[5]);
    int dpos = 0;
    data_buf[dpos++] = '[';
    for (int k = 0; k < S; k++) {
        dpos += snprintf(data_buf + dpos, sizeof(data_buf) - dpos,
                         k ? ",%d,%d" : "%d,%d", buf[2 * k], buf[2 * k + 1]);
    }
    data_buf[dpos++] = ']'; data_buf[dpos] = '\0';
    uint32_t ts_us = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFF);
    // One esp-csi CSV line; the batch task adds the JSON header. mac (col 2) carries the tx id;
    // local_timestamp (col 18) lets the host reconstruct each frame's absolute time within the batch.
    size_t cap = (size_t)dpos + 96;
    char *line = (char *)malloc(cap);
    if (!line) return;
    snprintf(line, cap,
             "CSI_DATA,0,%s,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,%u,0,0,0,%d,0,%s\n",
             mac, (unsigned)ts_us, S * 2, data_buf);
    if (xQueueSend(s_csi_q, &line, 0) != pdTRUE) free(line);
    s_csi_count++;
}

// Learn sender MACs + liveness, track airtime, and accept the token when a burst hands us the turn.
static void espnow_recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len < (int)sizeof(mesh_pkt_t)) return;
    const mesh_pkt_t *p = (const mesh_pkt_t *)data;
    if (p->magic != MESH_MAGIC) return;
    s_last_air_ms = now_ms();
    learn_mac(info->src_addr, p->tx_id);
    if ((p->flags & FLAG_LAST) && p->next_id == NODE_ID) {
        uint8_t go = 1;
        xQueueSend(s_turn_q, &go, 0);  // our turn next
    }
}

// Transmit one burst, handing the token to the next LIVE node id on the final frame.
static volatile uint32_t s_last_tx_ms = 0;

static void send_burst(void) {
    s_last_tx_ms = now_ms();
    uint8_t next = (uint8_t)next_after(NODE_ID);
    for (int seq = 0; seq < BURST_LEN; seq++) {
        mesh_pkt_t p = {MESH_MAGIC, (uint8_t)NODE_ID, (uint8_t)seq,
                        (uint8_t)(seq == BURST_LEN - 1 ? FLAG_LAST : 0), next};
        if (esp_now_send(BCAST, (uint8_t *)&p, sizeof(p)) == ESP_OK) s_tx_sent++;
        vTaskDelay(pdMS_TO_TICKS(BURST_MS));
    }
}

// Token loop. Whoever is the current leader (lowest live id) bootstraps and self-heals a lost token;
// everyone bursts when handed the turn. Leadership follows the ring as boards join/leave.
static void mesh_task(void *) {
    vTaskDelay(pdMS_TO_TICKS(3000));  // settle: STA assoc + hear peers before deciding the ring
    if (s_connected && NODE_ID == leader_id()) {
        ESP_LOGI(TAG, "bootstrap: i am leader, starting mesh");
        send_burst();
    }
    for (;;) {
        uint8_t go;
        if (xQueueReceive(s_turn_q, &go, pdMS_TO_TICKS(TURN_TIMEOUT_MS)) == pdTRUE) {
            if (s_connected) send_burst();
        } else if (s_connected) {
            uint32_t t = now_ms();
            if (NODE_ID == leader_id()) {
                if (t - s_last_air_ms > TURN_TIMEOUT_MS) {
                    ESP_LOGD(TAG, "token lost, restarting");
                    send_burst();
                }
            } else if (t - s_last_tx_ms > 5000) {
                // Discovery: if we haven't transmitted in 5s, send a single frame to introduce ourselves.
                // This lets the leader learn our MAC and include us in the ring.
                mesh_pkt_t p = {MESH_MAGIC, (uint8_t)NODE_ID, 0, 0, 0};
                esp_now_send(BCAST, (uint8_t *)&p, sizeof(p));
                s_last_tx_ms = t;
                ESP_LOGI(TAG, "sent discovery frame (peers=%d leader=%d)", alive_peers(), leader_id());
            }
        }
    }
}

// Batch CSI lines into one datagram per BATCH_MS (or UDP_BATCH_BYTES), instead of one datagram per
// frame — at ~300 fps per-frame UDP means thousands of contending TXs/s and heavy loss.
static void udp_batch_task(void *) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in dst = {};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(CSI_UDP_PORT);
    static char body[UDP_BATCH_BYTES];
    static char dgram[UDP_BATCH_BYTES + 128];
    int blen = 0, count = 0;
    char *line;
    for (;;) {
        bool flush = false;
        if (xQueueReceive(s_csi_q, &line, pdMS_TO_TICKS(BATCH_MS)) == pdTRUE) {
            int ll = (int)strlen(line);
            if (blen + ll >= UDP_BATCH_BYTES) flush = (count > 0);
            if (!flush && ll < UDP_BATCH_BYTES) {
                memcpy(body + blen, line, ll); blen += ll; count++;
            }
            char *pending = flush ? line : NULL;
            if (flush) {
                int hl = snprintf(dgram, sizeof(dgram),
                                  "{\"v\":1,\"node\":%d,\"ntp_ms\":%llu,\"n\":%d}\n",
                                  NODE_ID, (unsigned long long)wall_ms(), count);
                memcpy(dgram + hl, body, blen);
                dst.sin_addr = s_pc_addr;
                sendto(sock, dgram, hl + blen, 0, (struct sockaddr *)&dst, sizeof(dst));
                blen = 0; count = 0;
                int pl = (int)strlen(pending);
                if (pl < UDP_BATCH_BYTES) { memcpy(body + blen, pending, pl); blen += pl; count++; }
            }
            free(line);
        } else if (count > 0) {
            int hl = snprintf(dgram, sizeof(dgram),
                              "{\"v\":1,\"node\":%d,\"ntp_ms\":%llu,\"n\":%d}\n",
                              NODE_ID, (unsigned long long)wall_ms(), count);
            memcpy(dgram + hl, body, blen);
            dst.sin_addr = s_pc_addr;
            sendto(sock, dgram, hl + blen, 0, (struct sockaddr *)&dst, sizeof(dst));
            blen = 0; count = 0;
        }
    }
}

// Per-node heartbeat to the PC on HEALTH_UDP_PORT (separate from the CSI stream so the CSI parser
// never sees it). Lets the PC show per-node health even when CSI flow looks fine.
static void health_task(void *) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in dst = {};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(HEALTH_UDP_PORT);
    char msg[288];
    uint32_t last_csi = 0, last_tx = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(HEALTH_MS));
        wifi_ap_record_t ap; int rssi = 0;
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) rssi = ap.rssi;
        uint32_t c = s_csi_count, tx = s_tx_sent;
        const char *g = s_gain_locked == 1 ? "LOCK" : s_gain_locked == 2 ? "SKIP" : "coll";
        int n = snprintf(msg, sizeof(msg),
            "{\"v\":1,\"node\":%d,\"type\":\"health\",\"up_s\":%llu,\"heap\":%lu,\"rssi\":%d,"
            "\"csi_hz\":%lu,\"tx_hz\":%lu,\"peers\":%d,\"leader\":%d,\"gain\":\"%s\",\"agc\":%u,"
            "\"synced\":%d}\n",
            NODE_ID, (unsigned long long)(esp_timer_get_time() / 1000000),
            (unsigned long)esp_get_free_heap_size(), rssi,
            (unsigned long)((c - last_csi) * 1000 / HEALTH_MS),
            (unsigned long)((tx - last_tx) * 1000 / HEALTH_MS),
            alive_peers(), leader_id(), g, (unsigned)s_lock_agc, wall_ms() > 1600000000000ULL ? 1 : 0);
        dst.sin_addr = s_pc_addr;
        sendto(sock, msg, n, 0, (struct sockaddr *)&dst, sizeof(dst));
        last_csi = c; last_tx = tx;
    }
}

// 1 s serial heartbeat — the health view when there is NO PC (just a USB monitor).
static void stats_task(void *) {
    uint32_t last = 0, last_raw = 0, last_tx = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        uint32_t n = s_csi_count, raw = s_csi_raw, tx = s_tx_sent;
        wifi_ap_record_t ap; int rssi = 0;
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) rssi = ap.rssi;
        ESP_LOGI(TAG,
            "node=%d csi_hz=%lu csi_raw=%lu tx=%lu peers=%d leader=%d gain=%s(%u/%d) rssi=%d heap=%lu",
            NODE_ID, (unsigned long)(n - last), (unsigned long)(raw - last_raw),
            (unsigned long)(tx - last_tx), alive_peers(), leader_id(),
            s_gain_locked == 1 ? "LOCK" : s_gain_locked == 2 ? "SKIP" : "coll",
            (unsigned)s_lock_agc, (int)s_lock_fft, rssi, (unsigned long)esp_get_free_heap_size());
        last = n; last_raw = raw; last_tx = tx;
    }
}

#if STATUS_LED_GPIO >= 0
// Onboard WS2812 = the health view with NO PC and NO USB (just power). STATUS_LED_GPIO is the
// WS2812 data line (GPIO 38 on DevKitC-1 v1.1; the board powers the LED off the rail, no enable pin).
// RGB_PWR_GPIO stays undefined for these boards: defining it forces the data line HIGH and kills the LED.
static void led_task(void *) {
#ifdef RGB_PWR_GPIO
    esp_rom_gpio_pad_select_gpio(RGB_PWR_GPIO);
    gpio_set_direction((gpio_num_t)RGB_PWR_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level((gpio_num_t)RGB_PWR_GPIO, 1);
#endif
    led_strip_handle_t led = NULL;
    led_strip_config_t scfg = {
        .strip_gpio_num = STATUS_LED_GPIO,
        .max_leds = 1,
        .led_pixel_format = LED_PIXEL_FORMAT_GRB,
        .led_model = LED_MODEL_WS2812,
        .flags = { .invert_out = false },
    };
    led_strip_rmt_config_t rcfg = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .flags = { .with_dma = false },
    };
    if (led_strip_new_rmt_device(&scfg, &rcfg, &led) != ESP_OK) vTaskDelete(NULL);

    uint32_t last_csi = 0;
    for (;;) {
        uint32_t c = s_csi_count, d = c - last_csi; last_csi = c;
        int peers = alive_peers();
        uint8_t r = 0, g = 0, b = 0, br = 32;
        if (!s_connected)              { r = br; }            // RED: no Wi-Fi
        else if (s_gain_locked == 0)   { r = br; g = br/2; }  // YELLOW: calibrating
        else if (peers == 0)           { b = br; }            // BLUE: solo
        else if (d == 0)               { r = br; b = br; }    // MAGENTA: silent mesh
        else                           { g = br; }            // GREEN: healthy
        led_strip_set_pixel(led, 0, r, g, b);
        led_strip_refresh(led);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
#endif

// Promiscuous is enabled only so the CSI engine taps non-AP frames; the packets themselves go unused.
static void promisc_cb(void *buf, wifi_promiscuous_pkt_type_t type) {}

static void wifi_event_handler(void *, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        s_connected = false;   // stop bursting so the radio is free to re-associate fast
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        s_connected = true;
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "router ip: " IPSTR, IP2STR(&ev->ip_info.ip));
    }
}

extern "C" void app_main(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_t *netif = esp_netif_create_default_wifi_sta();
    char hostname[20];
    snprintf(hostname, sizeof(hostname), "wavetrace%d", NODE_ID);
    esp_netif_set_hostname(netif, hostname);

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL);
    esp_wifi_set_mode(WIFI_MODE_STA);
    wifi_config_t sta_cfg = {};
    strncpy((char *)sta_cfg.sta.ssid, ROUTER_SSID, sizeof(sta_cfg.sta.ssid));
    strncpy((char *)sta_cfg.sta.password, ROUTER_PASS, sizeof(sta_cfg.sta.password));
    esp_wifi_set_config(WIFI_IF_STA, &sta_cfg);
    esp_wifi_start();
    esp_wifi_set_ps(WIFI_PS_NONE);  // no modem-sleep -> steady bursts + reliable RX
    esp_wifi_connect();

    // mDNS: each node advertises as wavetraceN.local on every PC/Mac via Bonjour/Avahi
    mdns_init();
    mdns_hostname_set(hostname);

    // Shared wall clock (best-effort; the mesh runs regardless of whether this ever syncs).
    esp_sntp_config_t sntp = ESP_NETIF_SNTP_DEFAULT_CONFIG(SNTP_SERVER);
    esp_netif_sntp_init(&sntp);

    // ESP-NOW on the STA interface = the current (router) channel, so every node hears every other.
    esp_now_init();
    esp_now_register_recv_cb(espnow_recv_cb);
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BCAST, 6);
    peer.channel = 0;            // 0 = current channel (locked by the STA association)
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    // Force the sensing frame to an HT rate. Default ESP-NOW = tiny legacy/1 Mbps action frames that
    // rarely trigger the CSI engine (our yield was <1%); HT frames yield CSI reliably AND carry HT-LTF.
    // HT20 matches a 20 MHz AP. HT40 (wider CSI) is REJECTED on a BW20 channel ("invalid chanel info,
    // need change second channel to 40") and silently falls back to legacy — only use HT40 once the
    // router runs a 40 MHz channel.
    esp_now_rate_config_t rate_cfg = {
        .phymode = WIFI_PHY_MODE_HT20,
        .rate = WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false,
    };
    esp_err_t rerr = esp_now_set_peer_rate_config(BCAST, &rate_cfg);
    if (rerr != ESP_OK) ESP_LOGW(TAG, "esp_now rate_config failed: %s", esp_err_to_name(rerr));

    wifi_csi_config_t csi_cfg = {
        .lltf_en = true, .htltf_en = true, .stbc_htltf2_en = true,
        .ltf_merge_en = true, .channel_filter_en = false, .manu_scale = false,
    };
    esp_wifi_set_csi_config(&csi_cfg);
    esp_wifi_set_csi_rx_cb(csi_cb, NULL);
    esp_wifi_set_csi(true);

    // CSI only fires for the associated-AP link by default. Promiscuous mode taps EVERY received
    // frame so CSI is also generated from peers' ESP-NOW bursts. ESP-NOW = vendor action (mgmt)
    // frames, so MGMT must be in the filter; DATA kept too (harmless, also yields CSI).
    esp_wifi_set_promiscuous_rx_cb(promisc_cb);
    wifi_promiscuous_filter_t promisc_filt = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT | WIFI_PROMIS_FILTER_MASK_DATA,
    };
    esp_wifi_set_promiscuous_filter(&promisc_filt);
    esp_wifi_set_promiscuous(true);

    s_csi_q = xQueueCreate(64, sizeof(char *));   // deeper queue: ~BATCH_MS of lines buffer here
    s_turn_q = xQueueCreate(4, sizeof(uint8_t));
    xTaskCreate(discovery_task, "discovery", 3072, NULL, 4, NULL);
    xTaskCreate(udp_batch_task, "csi_udp", 4096, NULL, 5, NULL);
    xTaskCreate(mesh_task, "mesh", 4096, NULL, 6, NULL);
    xTaskCreate(health_task, "health", 4096, NULL, 3, NULL);
    xTaskCreate(stats_task, "stats", 4096, NULL, 2, NULL);
#if STATUS_LED_GPIO >= 0
    xTaskCreate(led_task, "led", 3072, NULL, 1, NULL);
#endif
}
