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

#define MESH_MAGIC 0x57            // 'W' — marks our mesh frames, ignore foreign ESP-NOW
#define FLAG_LAST  0x01            // set on the final frame of a burst (carries the token handoff)
// MTU-safe: keep the whole datagram (header + body) under one ~1500-byte Wi-Fi/Ethernet frame so a
// single dropped IP fragment can't sink the whole batch. Was 8192 (≈6 fragments -> amplified loss).
#define UDP_BATCH_BYTES 1400       // max CSI body per datagram before a forced flush
#define UDP_SEND_RETRIES 3         // resend a batch this many times on transient ENOMEM before dropping
#define UDP_ENOMEM_BACKOFF_MS 15   // brief global pause once the retries are exhausted (was a 100 ms hole)
// CSI byte length is DERIVED from the sensing mode (2 bytes/subcarrier, set by WT_BW_HT40 in
// config.h), not a magic number: HT40 = LLTF(64)+HT-LTF(128) = 192 complex = 384 B; HT20 =
// LLTF(64)+HT-LTF(64) = 128 complex = 256 B. The static by-value queue buffer is sized to exactly
// the active mode's width — a static array can't be sized at runtime, but the mode is known at build
// time. (The ACTUAL per-frame length still pins dynamically via s_expect_len; this is just the cap.)
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

// Raw CSI passed BY VALUE through the queue (no per-frame malloc, no formatting in the Wi-Fi task).
// udp_batch_task on Core 1 packs this into a binary v2 record (csi_hdr_t below).
typedef struct {
    uint8_t  mac[6];     // transmitter MAC = tx node identity
    uint16_t len;        // pinned CSI byte length (STBC doubles already collapsed)
    uint32_t ts_us;      // per-frame local esp_timer time (low 32 bits)
    int8_t   buf[CSI_MAX_BYTES];
} csi_raw_t;

// Binary UDP wire format v2 (little-endian; both ESP32 and the host Mac are LE -> no byteswap).
// One datagram = csi_hdr_t, then `n` packed records: mac[6] | ts_us(u32) | len(u16) | len*int8 raw CSI.
// ~3.5x smaller than the old ASCII CSV (140 vs ~500 B/frame @ HT20) and zero formatting cost.
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
static volatile uint32_t s_csi_raw = 0;   // DIAG: every CSI callback, before the MAC filter
static volatile uint32_t s_tx_sent = 0;   // DIAG: ESP-NOW frames we successfully queued for TX
static volatile uint32_t s_last_air_ms = 0;   // last time any mesh frame was heard
static volatile bool s_connected = false;     // STA associated + has IP (gate mesh TX on this)
static struct in_addr s_pc_addr = {0};        // discovered PC address
static volatile int s_expect_len = 0;         // first-seen CSI byte length; the stream is pinned to this

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
    return -1;  // not a mesh node -> not our sensing traffic
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
        if (s_miss[i] >= TOKEN_MISS_MAX) continue;  // adaptive skip: unresponsive to recent handoffs
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

// Capture CSI; copy the raw I/Q + tx identity into the queue BY VALUE. No malloc, no snprintf here —
// this runs in the Wi-Fi task (Core 0); heap ops + string formatting at ~300 fps would starve it.
static void csi_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) return;
    s_csi_raw++;  // DIAG: count every CSI callback regardless of source
    int tx = id_for_mac(info->mac);
    if (tx < 0) return;  // only emit CSI from known mesh nodes
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
    // Pin the CSI width at the source: the host gets a deterministic (1,S) shape, and STBC double-length
    // frames are RETAINED (collapsed to the first LTF block) instead of being dropped by the host guard.
    int len = info->len;
    if (len <= 0 || len > 2 * CSI_MAX_BYTES) return;
    if (s_expect_len == 0) {
        if (len > CSI_MAX_BYTES) return;            // wait for a base-width frame to pin to
        s_expect_len = len;
    }
    if (len == 2 * s_expect_len) len = s_expect_len;   // STBC double -> keep first LTF block only
    else if (len != s_expect_len) return;              // off-format (short/odd) -> drop, like the host

    // Per-node emit-rate cap (CSI_MAX_HZ): drop frames closer than the min interval so HT40's high
    // native rate can't overrun the UDP backhaul (sendto ENOMEM). The host resamples to TARGET_FS
    // anyway, so the dropped frames carry no usable signal. csi_raw still counts every callback, so
    // the gap between csi_raw and csi_hz in the heartbeat shows the cap working.

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
static volatile uint8_t  s_last_token_tx = 0;  // ...and from which tx node (token is repeated TOKEN_REPEAT x)
static volatile int  s_handoff_id = -1;        // node we last handed the token to (adaptive-skip pending)
static volatile bool s_handoff_heard = false;  // did that node transmit in response to our handoff?

// Learn sender MACs + liveness, track airtime, and accept the token when a burst hands us the turn.
static void espnow_recv_cb(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (len < (int)sizeof(mesh_pkt_t)) return;
    const mesh_pkt_t *p = (const mesh_pkt_t *)data;
    if (p->magic != MESH_MAGIC) return;
    s_last_air_ms = now_ms();
    learn_mac(info->src_addr, p->tx_id);
    if (p->tx_id == s_handoff_id) s_handoff_heard = true;  // it took its turn -> responsive
    int hi = idx_for_id(p->tx_id);
    if (hi >= 0) s_miss[hi] = 0;                           // hearing it transmit clears its skip count
    // The token rides the last TOKEN_REPEAT frames (survives single-frame loss); accept it ONCE per
    // burst by ignoring repeats from the same tx within TOKEN_DEDUP_MS, else we'd burst TOKEN_REPEAT x.
    if ((p->flags & FLAG_LAST) && p->next_id == NODE_ID) {
        uint32_t t = now_ms();
        if (!(p->tx_id == s_last_token_tx && t - s_last_token_ms < TOKEN_DEDUP_MS)) {
            s_last_token_tx = p->tx_id;
            s_last_token_ms = t;
            uint8_t go = 1;
            xQueueSend(s_turn_q, &go, 0);  // our turn next
        }
    }
}

// Transmit one burst, handing the token to the next LIVE node id on the final frame.
static volatile uint32_t s_last_tx_ms = 0;

static void send_burst(void) {
    s_last_tx_ms = now_ms();
    // Adaptive skip: judge the PREVIOUS handoff before issuing a new one. If the node we handed the
    // token to never transmitted (we didn't hear it) by the time the token came back to us, count a
    // miss; after TOKEN_MISS_MAX it's dropped from next_after immediately instead of lingering for
    // LIVE_TIMEOUT_MS (a flapping node would otherwise get — and drop — the token every lap). Any
    // frame we hear from it resets the count (espnow_recv_cb), so it rejoins as soon as it's responsive.
    if (s_handoff_id >= 0 && s_handoff_id != NODE_ID && !s_handoff_heard) {
        int hi = idx_for_id(s_handoff_id);
        if (hi >= 0 && s_miss[hi] < 255) s_miss[hi]++;
    }
    uint8_t next = (uint8_t)next_after(NODE_ID);
    s_handoff_id = (next != NODE_ID) ? next : -1;
    s_handoff_heard = false;
    for (int seq = 0; seq < BURST_LEN; seq++) {
        // Token (FLAG_LAST + next_id) is repeated on the last TOKEN_REPEAT frames so a single dropped
        // frame can't stall the ring; the receiver dedups (espnow_recv_cb) so it still bursts once.
        mesh_pkt_t p = {MESH_MAGIC, (uint8_t)NODE_ID, (uint8_t)seq,
                        (uint8_t)(seq >= BURST_LEN - TOKEN_REPEAT ? FLAG_LAST : 0), next};
        // Retry on NO_MEM (TX buffers full): the token frames matter most, so dropping one would
        // stall the ring until a TURN_TIMEOUT_MS self-heal. vTaskDelay(1) yields ~1 ms.
        esp_err_t e;
        for (int t = 0; (e = esp_now_send(BCAST, (uint8_t *)&p, sizeof(p))) == ESP_ERR_ESPNOW_NO_MEM && t < 3; t++)
            vTaskDelay(1);
        if (e == ESP_OK) s_tx_sent++;
        vTaskDelay(pdMS_TO_TICKS(BURST_MS));
    }
}

// Token loop. Whoever is the current leader (lowest live id) bootstraps and self-heals a lost token;
// everyone bursts when handed the turn. Leadership follows the ring as boards join/leave.
static volatile bool s_ever_got_turn = false;  // admitted to the ring once we've received >=1 token

static void mesh_task(void *) {
    vTaskDelay(pdMS_TO_TICKS(3000));  // settle: STA assoc + hear peers before deciding the ring
    if (s_connected && NODE_ID == leader_id()) {
        ESP_LOGI(TAG, "bootstrap: i am leader, starting mesh");
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
                // Backstop: the whole ring (incl. the leader) has been silent far longer than a turn.
                // Any node restarts it; the +id*100 stagger makes only the lowest live id fire first, so
                // a dead leader no longer freezes everyone (otherwise only the leader self-heals).
                ESP_LOGD(TAG, "leader dead, taking over");
                send_burst();
            } else {
                // Announce until admitted (got our first token), then back off to a 5 s keep-alive.
                // Aggressive DISCOVERY_MS announce so the ring learns a fresh joiner within ~1 lap
                // instead of the old fixed 5 s; other nodes learn_mac us -> next_after includes us.
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

// Prepend the binary header and ship one datagram (header + body). dst addr is refreshed per send.
static int send_csi_batch(int sock, struct sockaddr_in *dst, uint8_t *dgram,
                          const uint8_t *body, int blen, int count) {
    csi_hdr_t h = {MESH_MAGIC, 2, (uint8_t)NODE_ID, wall_ms(), (uint16_t)count};
    memcpy(dgram, &h, sizeof(h));
    memcpy(dgram + sizeof(h), body, blen);
    dst->sin_addr = s_pc_addr;
    return sendto(sock, dgram, sizeof(h) + blen, 0, (struct sockaddr *)dst, sizeof(*dst));
}

// Flush one batch with a bounded retry on transient ENOMEM (TX buffers momentarily full). Keeps the
// SAME bytes and yields ~2 ms between tries instead of discarding the datagram on the first failure —
// so one ENOMEM blip no longer becomes a multi-hundred-frame hole (the queue would overflow during a
// 100 ms drop). Returns true once it leaves the air (or on a non-ENOMEM error a retry can't fix);
// false only after every retry still hit ENOMEM -> caller backs off briefly then drops this batch.
static bool flush_csi_batch(int sock, struct sockaddr_in *dst, uint8_t *dgram,
                            const uint8_t *body, int blen, int count) {
    for (int t = 0; t < UDP_SEND_RETRIES; t++) {
        if (send_csi_batch(sock, dst, dgram, body, blen, count) >= 0) return true;
        if (errno != ENOMEM) return true;       // not a buffer issue -> retrying won't help
        vTaskDelay(pdMS_TO_TICKS(2));           // let the Wi-Fi TX path drain, then resend the SAME batch
    }
    return false;
}

// Pack raw CSI -> binary records and batch them into one datagram per BATCH_MS (or per MTU), instead
// of one datagram per frame — at ~300 fps per-frame UDP means thousands of contending TXs/s and heavy
// loss. Packing is pure memcpy (no snprintf), so Core 1 stays cheap and the Wi-Fi callback (Core 0) free.
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
                ESP_LOGI(TAG, "ENOMEM backoff expired, resuming sends");
                backoff_until_us = 0;
            }
        }

        if (xQueueReceive(s_csi_q, &r, pdMS_TO_TICKS(BATCH_MS)) == pdTRUE) {
            int rec = 6 + 4 + 2 + r.len;   // mac | ts_us | len | raw int8 CSI
            if (blen + rec > UDP_BATCH_BYTES && count > 0) {
                if (!flush_csi_batch(sock, &dst, dgram, body, blen, count)) {
                    backoff_until_us = esp_timer_get_time() + UDP_ENOMEM_BACKOFF_MS * 1000;
                    ESP_LOGW(TAG, "sendto ENOMEM after %d retries — backing off %d ms",
                             UDP_SEND_RETRIES, UDP_ENOMEM_BACKOFF_MS);
                }
                blen = 0; count = 0;   // batch left the air (or dropped after retries) -> reset
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
            // When the queue stays full (busy AP) this loop never blocks on xQueueReceive and would
            // starve the Core-1 idle task (task-WDT) + the other pinned tasks (health/mesh). Yield ~1ms
            // every 64 frames so they run; at >1000 fps this costs <2% throughput.
            if (++since_yield >= 64) { since_yield = 0; vTaskDelay(1); }
        } else if (count > 0) {
            if (!flush_csi_batch(sock, &dst, dgram, body, blen, count)) {
                backoff_until_us = esp_timer_get_time() + UDP_ENOMEM_BACKOFF_MS * 1000;
                ESP_LOGW(TAG, "sendto ENOMEM after %d retries — backing off %d ms",
                         UDP_SEND_RETRIES, UDP_ENOMEM_BACKOFF_MS);
            }
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
        int res = sendto(sock, msg, n, 0, (struct sockaddr *)&dst, sizeof(dst));
        if (res < 0) {
            ESP_LOGW(TAG, "health sendto failed: errno %d", errno);
        }
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
        // HT-rate the ESP-NOW sensing burst HERE (post-association), NOT in app_main: only once we are
        // on the router channel does the 40 MHz secondary exist. Setting an HT40 rate before assoc (still
        // on the default 20 MHz channel) fails with "invalid chanel info / ESP_ERR_ESPNOW_ARG" and the
        // burst silently stays legacy → CSI is HT20-width even on a 40 MHz link. Re-applied each (re)assoc.
        esp_now_rate_config_t rate_cfg = {
            .phymode = WT_BW_HT40 ? WIFI_PHY_MODE_HT40 : WIFI_PHY_MODE_HT20,
            .rate = WIFI_PHY_RATE_MCS0_LGI,
            .ersu = false,
            .dcm = false,
        };
        esp_err_t rerr = esp_now_set_peer_rate_config(BCAST, &rate_cfg);
        ESP_LOGI(TAG, "esp_now HT%d rate: %s", WT_BW_HT40 ? 40 : 20, esp_err_to_name(rerr));

        // DIAGNOSTIC: is the router actually on a 40 MHz channel? `second` is read from the ASSOCIATED
        // AP (ground truth, not what we requested). second != NONE => the AP offers a 40 MHz secondary;
        // second == NONE while WT_BW_HT40=1 => the router is 20 MHz-only, so HT40 ESP-NOW frames won't
        // carry and peers/csi stay 0. Printed at WARN so it stands out once per (re)association.
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
    // Lock the PHY so the AP can't push us into 11ax/HT40 (which would change the CSI subcarrier layout)
    // or a bandwidth that conflicts with the ESP-NOW sensing rate. WT_BW_HT40 selects the bandwidth;
    // HT40 needs the router on a 40 MHz channel (else ESP-NOW HT40 silently drops to legacy).
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
    esp_wifi_set_bandwidth(WIFI_IF_STA, WT_BW_HT40 ? WIFI_BW_HT40 : WIFI_BW_HT20);
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

    // The sensing frame is forced to an HT rate (default ESP-NOW = tiny legacy/1 Mbps action frames that
    // rarely trigger the CSI engine, <1% yield; HT frames yield CSI reliably AND carry HT-LTF). The
    // rate_config is applied in wifi_event_handler on IP_EVENT_STA_GOT_IP — NOT here — because an HT40
    // rate needs the 40 MHz secondary channel, which only exists after we associate to the router.

    // Promiscuous mode is set up BEFORE the CSI engine (ESPectre ordering): it primes the internal
    // Wi-Fi stack tables so CSI initializes cleanly. CSI only fires for the associated-AP link by
    // default; promiscuous taps received frames so CSI is also generated from peers' ESP-NOW bursts.
    // ESP-NOW = vendor action (MGMT) frames, so MGMT alone captures ALL our sensing traffic. DATA is
    // excluded on purpose: a busy AP's data flood drove 100s of HW interrupts/s and corrupted the
    // sniffer RX buffers (wDev_SnifferRxData LoadProhibited crash-loop); those frames are never used.
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

    // By-value queue: 128 frames ≈ 0.3–0.5 s of burst tolerance — ample vs the 100 ms BATCH_MS flush.
    // Static cost at HT40 = 128*sizeof(csi_raw_t) (≈396 B) ≈ 50 KB. Was 256 (≈99 KB at HT40, NOT the
    // 68 KB the old comment claimed — that was the HT20 figure); the oversize starved the heap that the
    // Wi-Fi/lwIP TX path needs, *causing* the sendto ENOMEM it was meant to buffer against. Halving it
    // frees ~50 KB for the (now static) TX buffers. (No per-frame heap churn — the malloc was a
    // CLAUDE.md hot-path-allocation violation.)
    s_csi_q = xQueueCreate(128, sizeof(csi_raw_t));
    s_turn_q = xQueueCreate(4, sizeof(uint8_t));
    // Pin every app task to Core 1 (APP_CPU), leaving Core 0 (PRO_CPU) dedicated to the Wi-Fi/lwIP
    // stack + csi_cb. The binary record packing in udp_batch_task (pure memcpy) must not contend with
    // the driver.
    xTaskCreatePinnedToCore(discovery_task, "discovery", 3072, NULL, 4, NULL, 1);
    xTaskCreatePinnedToCore(udp_batch_task, "csi_udp", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(mesh_task, "mesh", 4096, NULL, 6, NULL, 1);
    xTaskCreatePinnedToCore(health_task, "health", 4096, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(stats_task, "stats", 4096, NULL, 2, NULL, 1);
#if STATUS_LED_GPIO >= 0
    xTaskCreatePinnedToCore(led_task, "led", 3072, NULL, 1, NULL, 1);
#endif
}
