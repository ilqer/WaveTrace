// Mesh node: STA (backhaul) + ESP-NOW (sensing).
// Time-division round-robin: one transmits, others capture CSI. Rotating N*(N-1) links/cycle.
// Dynamic ring: nodes learn peers via liveness timeout. Leader is lowest live ID. Token passes to next-higher ID on burst's last frame.

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <errno.h>
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

#define MESH_MAGIC 0x57            // 'W', marks our mesh frames so foreign ESP-NOW traffic is ignored
#define FLAG_LAST  0x01            // set on a burst's final frame; carries the token handoff
// Keep datagram under 1500B to avoid multi-fragment IP drops.
#define UDP_BATCH_BYTES 1400       // max CSI body per datagram before a forced flush
#define UDP_SEND_RETRIES 3         // resend a batch this many times on transient ENOMEM before dropping
#define UDP_ENOMEM_BACKOFF_MS 15   // brief global pause once retries are exhausted (was a 100 ms hole)
// HT40 = 384B, HT20 = 256B. s_expect_len pins runtime frame length.
#if WT_BW_HT40
#define CSI_MAX_BYTES   384
#else
#define CSI_MAX_BYTES   256
#endif

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

// Pass raw CSI by value. No malloc or formatting in Wi-Fi task.
typedef struct {
    uint8_t  mac[6];     // transmitter MAC = tx node identity
    uint16_t len;        // pinned CSI byte length (STBC doubles already collapsed)
    uint32_t ts_us;      // per-frame local esp_timer time (low 32 bits)
    int8_t   buf[CSI_MAX_BYTES];
} csi_raw_t;

// v2 LE binary UDP format: csi_hdr_t + n records (mac|ts_us|len|CSI).
typedef struct __attribute__((packed)) {
    uint8_t  magic;      // MESH_MAGIC ('W')
    uint8_t  ver;        // 2 = binary
    uint8_t  node;       // rx node id
    uint64_t ntp_ms;     // batch send wall time (~ last frame's time; host reconstructs per-frame)
    uint16_t n;          // record count
} csi_hdr_t;             // 13 bytes

static QueueHandle_t s_csi_q;        // csi_raw_t frames -> udp_batch_task
static QueueHandle_t s_turn_q;       // "go" tokens -> mesh_task
static volatile uint32_t s_csi_count = 0;
static volatile uint32_t s_csi_raw = 0;   // diag: every CSI callback, before the MAC filter
static volatile uint32_t s_tx_sent = 0;   // diag: ESP-NOW frames successfully queued for TX
static volatile uint32_t s_last_air_ms = 0;   // last time any mesh frame was heard
static volatile bool s_connected = false;     // STA associated + has IP; gates mesh TX
static struct in_addr s_pc_addr = {0};        // discovered PC address
static volatile int s_expect_len = 0;         // first-seen CSI byte length; the stream pins to this

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
    inet_pton(AF_INET, PC_IP, &s_pc_addr); // fallback default
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

// PHY gain lock: collect AGC/FFT baseline, then force it in-chip for consistent CSI amplitude.
#define GAIN_BASELINE_PKTS 300
static volatile uint32_t s_gain_samples = 0;
static volatile int s_gain_locked = 0;        // diag: 0=collecting baseline, 1=gain forced, 2=skipped
static volatile uint8_t s_lock_agc = 0;       // diag: the locked AGC/FFT values
static volatile int8_t s_lock_fft = 0;

// Maps MAC to node ID and liveness from ESP-NOW frames.
static uint8_t s_macs[MAX_NODES][6];
static uint8_t s_ids[MAX_NODES];
static volatile uint32_t s_seen_ms[MAX_NODES];
static uint8_t s_miss[MAX_NODES];    // consecutive turns we handed this peer the token without hearing it
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
    return -1;  // not a mesh node, so not our sensing traffic
}

static int idx_for_id(int id) {
    for (int i = 0; i < s_nmac; i++) if (s_ids[i] == id) return i;
    return -1;
}

// --- Dynamic ring helpers (self is always alive; peers expire after LIVE_TIMEOUT_MS) ---
static int alive_peers(void) {
    uint32_t t = now_ms(); int c = 0;
    for (int i = 0; i < s_nmac; i++)
        if (s_ids[i] != NODE_ID && t - s_seen_ms[i] < LIVE_TIMEOUT_MS) c++;
    return c;
}
// Lowest live id, including self, is the leader.
static int leader_id(void) {
    uint32_t t = now_ms(); int best = NODE_ID;
    for (int i = 0; i < s_nmac; i++)
        if (t - s_seen_ms[i] < LIVE_TIMEOUT_MS && s_ids[i] < best) best = s_ids[i];
    return best;
}
// Next live id strictly greater than `me`, else wraps to the lowest live id. Self is a candidate.
static int next_after(int me) {
    uint32_t t = now_ms();
    int best_gt = 0x7fffffff, best_any = NODE_ID;
    for (int i = 0; i < s_nmac; i++) {
        if (t - s_seen_ms[i] >= LIVE_TIMEOUT_MS) continue;
        if (s_miss[i] >= TOKEN_MISS_MAX) continue;  // adaptive skip: unresponsive to recent handoff
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
    if (tv.tv_sec > 1600000000)  // plausible real time, so SNTP is synced
        return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
    return (uint64_t)(esp_timer_get_time() / 1000);
}

// csi_cb runs on Core 0 at ~300fps. No malloc/formatting allowed.
static void csi_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) return;
    s_csi_raw++;  // diag: count every CSI callback regardless of source
    int tx = id_for_mac(info->mac);
    if (tx < 0) return;  // only emit CSI from known mesh nodes
#if WIFI_CSI_PHY_GAIN_ENABLE
    // Freeze AGC/FFT once a baseline is built. Room needs to stay still for the first few seconds.
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
                // Skip lock if AGC < 30 (signal too strong, forcing causes WDT crash). Rely on host CV normalization.
                if (bagc < 30) {
                    s_gain_locked = 2;  // SKIP: too strong
                } else {
                    esp_csi_gain_ctrl_set_rx_force_gain(bagc, bfft);
                    s_gain_locked = 1;  // LOCK
                }
            } else {
                s_gain_samples = 0;  // baseline not ready yet, recollect
            }
        }
    }
#endif
    // Pin CSI width to keep (1,S) shape and collapse STBC frames instead of dropping them.
    int len = info->len;
    if (len <= 0 || len > 2 * CSI_MAX_BYTES) return;
    if (s_expect_len == 0) {
        if (len > CSI_MAX_BYTES) return;            // wait for a base-width frame to pin to
        s_expect_len = len;
    }
    if (len == 2 * s_expect_len) len = s_expect_len;   // STBC double, keep first LTF block only
    else if (len != s_expect_len) return;              // off-format (short/odd), drop like the host

    // Drop frames exceeding CSI_MAX_HZ to prevent UDP sendto ENOMEM overruns.
#if CSI_MAX_HZ > 0
    static int64_t s_last_emit_us = 0;
    const int64_t now_us = esp_timer_get_time();
    if (now_us - s_last_emit_us < 1000000 / CSI_MAX_HZ) return;
    s_last_emit_us = now_us;
#endif

    csi_raw_t r;
    memcpy(r.mac, info->mac, 6);
    r.len = (uint16_t)len;
    r.ts_us = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFF);
    memcpy(r.buf, info->buf, len);
    if (xQueueSend(s_csi_q, &r, 0) == pdTRUE) s_csi_count++;
}

static volatile uint32_t s_last_token_ms = 0;  // dedup: when we last accepted a token...
static volatile uint8_t  s_last_token_tx = 0;  // ...and from which tx node (token repeats TOKEN_REPEAT x)
static volatile int  s_handoff_id = -1;        // node we last handed the token to (adaptive-skip pending)
static volatile bool s_handoff_heard = false;  // did that node transmit in response to our handoff?

// Update liveness, accept token on burst handoff.
static void espnow_recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len < (int)sizeof(mesh_pkt_t)) return;
    const mesh_pkt_t *p = (const mesh_pkt_t *)data;
    if (p->magic != MESH_MAGIC) return;
    s_last_air_ms = now_ms();
    learn_mac(info->src_addr, p->tx_id);
    if (p->tx_id == s_handoff_id) s_handoff_heard = true;  // it took its turn, so it's responsive
    int hi = idx_for_id(p->tx_id);
    if (hi >= 0) s_miss[hi] = 0;                           // hearing it transmit clears its skip count
    // Token rides last TOKEN_REPEAT frames. Dedup within TOKEN_DEDUP_MS to prevent multi-bursts.
    if ((p->flags & FLAG_LAST) && p->next_id == NODE_ID) {
        uint32_t t = now_ms();
        if (!(p->tx_id == s_last_token_tx && t - s_last_token_ms < TOKEN_DEDUP_MS)) {
            s_last_token_tx = p->tx_id;
            s_last_token_ms = t;
            uint8_t go = 1;
            xQueueSend(s_turn_q, &go, 0);  // it's our turn next
        }
    }
}

// Transmit one burst, handing the token to the next LIVE node id on the final frame.
static volatile uint32_t s_last_tx_ms = 0;

static void send_burst(void) {
    s_last_tx_ms = now_ms();
    // Adaptive skip: drop unresponsive node after TOKEN_MISS_MAX misses instead of waiting for timeout.
    if (s_handoff_id >= 0 && s_handoff_id != NODE_ID && !s_handoff_heard) {
        int hi = idx_for_id(s_handoff_id);
        if (hi >= 0 && s_miss[hi] < 255) s_miss[hi]++;
    }
    uint8_t next = (uint8_t)next_after(NODE_ID);
    s_handoff_id = (next != NODE_ID) ? next : -1;
    s_handoff_heard = false;
    for (int seq = 0; seq < BURST_LEN; seq++) {
        // Repeat token on last TOKEN_REPEAT frames to survive drops.
        mesh_pkt_t p = {MESH_MAGIC, (uint8_t)NODE_ID, (uint8_t)seq,
                        (uint8_t)(seq >= BURST_LEN - TOKEN_REPEAT ? FLAG_LAST : 0), next};
        // Retry token frames on TX NO_MEM to avoid ring stall.
        esp_err_t e;
        for (int t = 0; (e = esp_now_send(BCAST, (uint8_t *)&p, sizeof(p))) == ESP_ERR_ESPNOW_NO_MEM && t < 3; t++)
            vTaskDelay(1);
        if (e == ESP_OK) s_tx_sent++;
        vTaskDelay(pdMS_TO_TICKS(BURST_MS));
    }
}

// Token loop: leader bootstraps/heals token. Nodes burst when handed turn.
static volatile bool s_ever_got_turn = false;  // admitted to the ring once we've received >=1 token

static void mesh_task(void *) {
    vTaskDelay(pdMS_TO_TICKS(3000));  // settle: let STA associate and hear peers before deciding the ring
    if (s_connected && NODE_ID == leader_id()) {
        ESP_LOGI(TAG, "bootstrap: leader, starting mesh");
        send_burst();
    }
    for (;;) {
        uint8_t go;
        if (xQueueReceive(s_turn_q, &go, pdMS_TO_TICKS(TURN_TIMEOUT_MS)) == pdTRUE) {
            s_ever_got_turn = true;  // we're in the ring
            if (s_connected) send_burst();
        } else if (s_connected) {
            uint32_t t = now_ms();
            if (NODE_ID == leader_id()) {
                if (t - s_last_air_ms > TURN_TIMEOUT_MS) {
                    ESP_LOGD(TAG, "token lost, restarting");
                    send_burst();
                }
            } else if (t - s_last_air_ms > LEADER_DEAD_MS + (uint32_t)NODE_ID * 100) {
                // Dead ring backstop: restart ring with id*100 stagger so lowest ID fires first.
                ESP_LOGD(TAG, "leader dead, taking over");
                send_burst();
            } else {
                // Send fast keep-alive until admitted, then slow to 5s.
                uint32_t interval = s_ever_got_turn ? 5000 : DISCOVERY_MS;
                if (t - s_last_tx_ms > interval) {
                    mesh_pkt_t p = {MESH_MAGIC, (uint8_t)NODE_ID, 0, 0, 0};
                    esp_now_send(BCAST, (uint8_t *)&p, sizeof(p));
                    s_last_tx_ms = t;
                }
            }
        }
    }
}

// Prepend header and ship datagram.
static int send_csi_batch(int sock, struct sockaddr_in *dst, uint8_t *dgram,
                          const uint8_t *body, int blen, int count) {
    csi_hdr_t h = {MESH_MAGIC, 2, (uint8_t)NODE_ID, wall_ms(), (uint16_t)count};
    memcpy(dgram, &h, sizeof(h));
    memcpy(dgram + sizeof(h), body, blen);
    dst->sin_addr = s_pc_addr;
    return sendto(sock, dgram, sizeof(h) + blen, 0, (struct sockaddr *)dst, sizeof(*dst));
}

// Flush batch. Retry transient ENOMEM to prevent queue overflow.
static bool flush_csi_batch(int sock, struct sockaddr_in *dst, uint8_t *dgram,
                            const uint8_t *body, int blen, int count) {
    for (int t = 0; t < UDP_SEND_RETRIES; t++) {
        if (send_csi_batch(sock, dst, dgram, body, blen, count) >= 0) return true;
        if (errno != ENOMEM) return true;       // not a buffer issue, so retrying won't help
        vTaskDelay(pdMS_TO_TICKS(2));           // let the Wi-Fi TX path drain, then resend the same batch
    }
    return false;
}

// Pack CSI into batch datagrams.
static void udp_batch_task(void *) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in dst = {};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(CSI_UDP_PORT);
    static uint8_t body[UDP_BATCH_BYTES];
    static uint8_t dgram[sizeof(csi_hdr_t) + UDP_BATCH_BYTES];
    int blen = 0, count = 0, since_yield = 0;
    csi_raw_t r;
    int64_t backoff_until_us = 0;
    for (;;) {
        int64_t now = esp_timer_get_time();
        if (backoff_until_us > 0) {
            if (now < backoff_until_us) {
                vTaskDelay(pdMS_TO_TICKS(5));
                continue;
            } else {
                ESP_LOGI(TAG, "ENOMEM backoff done, resuming sends");
                backoff_until_us = 0;
            }
        }

        if (xQueueReceive(s_csi_q, &r, pdMS_TO_TICKS(BATCH_MS)) == pdTRUE) {
            int rec = 6 + 4 + 2 + r.len;   // mac | ts_us | len | raw int8 CSI
            if (blen + rec > UDP_BATCH_BYTES && count > 0) {
                if (!flush_csi_batch(sock, &dst, dgram, body, blen, count)) {
                    backoff_until_us = esp_timer_get_time() + UDP_ENOMEM_BACKOFF_MS * 1000;
                    ESP_LOGW(TAG, "sendto ENOMEM after %d retries, backing off %d ms",
                             UDP_SEND_RETRIES, UDP_ENOMEM_BACKOFF_MS);
                }
                blen = 0; count = 0;   // batch left the air (or dropped after retries), so reset
            }
            if (rec <= UDP_BATCH_BYTES) {
                uint8_t *p = body + blen;
                memcpy(p, r.mac, 6);          p += 6;
                memcpy(p, &r.ts_us, 4);        p += 4;   // LE u32
                uint16_t L = r.len;
                memcpy(p, &L, 2);              p += 2;   // LE u16
                memcpy(p, r.buf, r.len);
                blen += rec; count++;
            }
            // Yield ~1ms every 64 frames to prevent Core-1 starvation if queue is full.
            if (++since_yield >= 64) { since_yield = 0; vTaskDelay(1); }
        } else if (count > 0) {
            if (!flush_csi_batch(sock, &dst, dgram, body, blen, count)) {
                backoff_until_us = esp_timer_get_time() + UDP_ENOMEM_BACKOFF_MS * 1000;
                ESP_LOGW(TAG, "sendto ENOMEM after %d retries, backing off %d ms",
                         UDP_SEND_RETRIES, UDP_ENOMEM_BACKOFF_MS);
            }
            blen = 0; count = 0;
        }
    }
}

// PC heartbeat via HEALTH_UDP_PORT.
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
        int res = sendto(sock, msg, n, 0, (struct sockaddr *)&dst, sizeof(dst));
        if (res < 0) {
            ESP_LOGW(TAG, "health sendto failed: errno %d", errno);
        }
        last_csi = c; last_tx = tx;
    }
}

// Serial heartbeat for USB monitor.
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
// WS2812 status LED. Do not define RGB_PWR_GPIO or it kills the LED.
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
        if (!s_connected)              { r = br; }            // red: no Wi-Fi
        else if (s_gain_locked == 0)   { r = br; g = br/2; }  // yellow: calibrating
        else if (peers == 0)           { b = br; }            // blue: solo
        else if (d == 0)               { r = br; b = br; }    // magenta: silent mesh
        else                           { g = br; }            // green: healthy
        led_strip_set_pixel(led, 0, r, g, b);
        led_strip_refresh(led);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
#endif

// Promiscuous mode is enabled only so the CSI engine taps non-AP frames; the packets go unused.
static void promisc_cb(void *buf, wifi_promiscuous_pkt_type_t type) {}

static void wifi_event_handler(void *, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        s_connected = false;   // stop bursting so the radio is free to re-associate fast
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        s_connected = true;
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "router ip: " IPSTR, IP2STR(&ev->ip_info.ip));
        // Set HT rate post-association since 40MHz secondary channel only exists then.
        esp_now_rate_config_t rate_cfg = {
            .phymode = WT_BW_HT40 ? WIFI_PHY_MODE_HT40 : WIFI_PHY_MODE_HT20,
            .rate = WIFI_PHY_RATE_MCS0_LGI,
            .ersu = false,
            .dcm = false,
        };
        esp_err_t rerr = esp_now_set_peer_rate_config(BCAST, &rate_cfg);
        ESP_LOGI(TAG, "esp_now HT%d rate: %s", WT_BW_HT40 ? 40 : 20, esp_err_to_name(rerr));

        // Warning if router is 20MHz-only while WT_BW_HT40=1.
        wifi_ap_record_t ap = {};
        wifi_bandwidth_t bw = WIFI_BW_HT20;
        esp_wifi_get_bandwidth(WIFI_IF_STA, &bw);
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            const char *sec = ap.second == WIFI_SECOND_CHAN_ABOVE ? "ABOVE(=40MHz)"
                            : ap.second == WIFI_SECOND_CHAN_BELOW ? "BELOW(=40MHz)"
                            : "NONE(=20MHz only)";
            ESP_LOGW(TAG, "AP-CHECK ssid=%s ch=%d second=%s 11n=%d rssi=%d sta_bw=%s", ap.ssid,
                     ap.primary, sec, ap.phy_11n, ap.rssi, bw == WIFI_BW_HT40 ? "HT40" : "HT20");
        }
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
    // Lock PHY to prevent AP pushing 11ax or incompatible bandwidths.
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
    esp_wifi_set_bandwidth(WIFI_IF_STA, WT_BW_HT40 ? WIFI_BW_HT40 : WIFI_BW_HT20);
    esp_wifi_set_ps(WIFI_PS_NONE);  // disable modem-sleep for steady bursts and reliable RX
    esp_wifi_connect();

    // mDNS: each node advertises as wavetraceN.local to every PC/Mac via Bonjour/Avahi
    mdns_init();
    mdns_hostname_set(hostname);

    // Shared wall clock, best-effort; the mesh runs regardless of whether this ever syncs.
    esp_sntp_config_t sntp = ESP_NETIF_SNTP_DEFAULT_CONFIG(SNTP_SERVER);
    esp_netif_sntp_init(&sntp);

    // ESP-NOW on the STA interface uses the current (router) channel, so every node hears every other.
    esp_now_init();
    esp_now_register_recv_cb(espnow_recv_cb);
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, BCAST, 6);
    peer.channel = 0;            // 0 = current channel (locked by the STA association)
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;
    esp_now_add_peer(&peer);

    // Use HT rate for sensing frames to trigger CSI reliably.

    // Promiscuous mode allows tapping peer ESP-NOW bursts. Filter DATA frames to prevent RX buffer crashes.
    esp_wifi_set_promiscuous_rx_cb(promisc_cb);
    wifi_promiscuous_filter_t promisc_filt = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT,
    };
    esp_wifi_set_promiscuous_filter(&promisc_filt);
    esp_wifi_set_promiscuous(true);

    wifi_csi_config_t csi_cfg = {
        .lltf_en = true, .htltf_en = true, .stbc_htltf2_en = true,
        .ltf_merge_en = true, .channel_filter_en = false, .manu_scale = false,
    };
    esp_wifi_set_csi_config(&csi_cfg);
    esp_wifi_set_csi_rx_cb(csi_cb, NULL);
    esp_wifi_set_csi(true);

    // 128-frame by-value queue (~50KB) prevents TX heap starvation.
    s_csi_q = xQueueCreate(128, sizeof(csi_raw_t));
    s_turn_q = xQueueCreate(4, sizeof(uint8_t));
    // Pin every app task to Core 1 (APP_CPU), leaving Core 0 (PRO_CPU) dedicated to the Wi-Fi/lwIP stack and csi_cb.
    xTaskCreatePinnedToCore(discovery_task, "discovery", 3072, NULL, 4, NULL, 1);
    xTaskCreatePinnedToCore(udp_batch_task, "csi_udp", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(mesh_task, "mesh", 4096, NULL, 6, NULL, 1);
    xTaskCreatePinnedToCore(health_task, "health", 4096, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(stats_task, "stats", 4096, NULL, 2, NULL, 1);
#if STATUS_LED_GPIO >= 0
    xTaskCreatePinnedToCore(led_task, "led", 3072, NULL, 1, NULL, 1);
#endif
}
