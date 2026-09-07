// Mesh node: STA (backhaul) + ESP-NOW (sensing). Time-division round-robin over N*(N-1) links/cycle.
// Dynamic ring: peers found via liveness timeout, leader = lowest live ID, token passes on last frame.

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
// HT40 = 384B, HT20 = 256B. sExpectedCsiLengthBytes pins runtime frame length.
#if WT_BW_HT40
#define CSI_MAX_BYTES   384
#else
#define CSI_MAX_BYTES   256
#endif

static const char *sTag = "wt_node";
static const uint8_t BROADCAST_MAC_ADDRESS[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

// Token frame: 5 bytes, broadcast. next_id on the LAST frame hands the turn to that node.
typedef struct __attribute__((packed)) {
    uint8_t magic;
    uint8_t tx_id;
    uint8_t seq;
    uint8_t flags;
    uint8_t next_id;
} MeshPacket;
static_assert(sizeof(MeshPacket) == 5, "wire format changed");

// Pass raw CSI by value. No malloc or formatting in Wi-Fi task.
typedef struct {
    uint8_t  mac[6];     // transmitter MAC = tx node identity
    uint16_t len;        // pinned CSI byte length (STBC doubles already collapsed)
    uint32_t ts_us;      // per-frame local esp_timer time (low 32 bits)
    int8_t   buf[CSI_MAX_BYTES];
} CsiRawFrame;

// v2 LE binary UDP format: CsiBatchHeader + n records (mac|ts_us|len|CSI).
typedef struct __attribute__((packed)) {
    uint8_t  magic;      // MESH_MAGIC ('W')
    uint8_t  ver;        // 2 = binary
    uint8_t  node;       // rx node id
    uint64_t ntp_ms;     // batch send wall time (~ last frame's time; host reconstructs per-frame)
    uint16_t n;          // record count
} CsiBatchHeader;        // 13 bytes
static_assert(sizeof(CsiBatchHeader) == 13, "wire format changed");

static QueueHandle_t sCsiFrameQueue;        // CsiRawFrame frames -> UdpBatchTask
static QueueHandle_t sTurnQueue;            // "go" tokens -> MeshTask
static volatile uint32_t sCsiFrameCount = 0;
static volatile uint32_t sCsiCallbackCount = 0;   // diag: every CSI callback, before the MAC filter
static volatile uint32_t sTxSentCount = 0;   // diag: ESP-NOW frames successfully queued for TX
static volatile uint32_t sLastAirTimeMs = 0;   // last time any mesh frame was heard
static volatile bool sbConnected = false;     // STA associated + has IP; gates mesh TX
static struct in_addr sPcAddress = {0};        // discovered PC address
static volatile int sExpectedCsiLengthBytes = 0;         // first-seen CSI byte length; the stream pins to this

static inline uint32_t CurrentTimeMs(void) { return (uint32_t)(esp_timer_get_time() / 1000); }

// PC discovery: listen for "WAVETRACE_PING" broadcast from health_monitor.py to find its IP.
static void DiscoveryTask(void *) {
    int discoverySocket = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in bindAddress = {};
    bindAddress.sin_family = AF_INET;
    bindAddress.sin_port = htons(DISCOVERY_PORT);
    bindAddress.sin_addr.s_addr = htonl(INADDR_ANY);
    bind(discoverySocket, (struct sockaddr *)&bindAddress, sizeof(bindAddress));
    char pingBuffer[32];
    inet_pton(AF_INET, PC_IP, &sPcAddress); // fallback default
    for (;;) {
        struct sockaddr_in senderAddress;
        socklen_t senderAddressLength = sizeof(senderAddress);
        int bytesReceived = recvfrom(discoverySocket, pingBuffer, sizeof(pingBuffer) - 1, 0, (struct sockaddr *)&senderAddress, &senderAddressLength);
        if (bytesReceived > 0) {
            pingBuffer[bytesReceived] = '\0';
            if (strstr(pingBuffer, "WAVETRACE_PING")) {
                if (senderAddress.sin_addr.s_addr != sPcAddress.s_addr) {
                    sPcAddress = senderAddress.sin_addr;
                    ESP_LOGI(sTag, "discovered pc at %s", inet_ntoa(sPcAddress));
                    esp_sntp_stop();
                    esp_sntp_setservername(0, inet_ntoa(sPcAddress));
                    esp_sntp_init();
                }
            }
        }
    }
}

// PHY gain lock: collect AGC/FFT baseline, then force it in-chip for consistent CSI amplitude.
#define GAIN_BASELINE_PACKET_COUNT 300
static volatile uint32_t sGainBaselineSampleCount = 0;
static volatile int sGainLockState = 0;        // diag: 0=collecting baseline, 1=gain forced, 2=skipped
static volatile uint8_t sLockedAgcValue = 0;       // diag: the locked AGC/FFT values
static volatile int8_t sLockedFftValue = 0;

// Maps MAC to node ID and liveness from ESP-NOW frames.
static uint8_t sPeerMacAddresses[MAX_NODES][6];
static uint8_t sPeerNodeIds[MAX_NODES];
static volatile uint32_t sPeerLastSeenMs[MAX_NODES];
static uint8_t sPeerMissedTurnCount[MAX_NODES];    // consecutive turns we handed this peer the token without hearing it
static volatile int sPeerCount = 0;

static void LearnPeerMacAddress(const uint8_t *peerMacAddress, uint8_t peerNodeId) {
    uint32_t nowMs = CurrentTimeMs();
    for (int i = 0; i < sPeerCount; i++) if (sPeerNodeIds[i] == peerNodeId) { sPeerLastSeenMs[i] = nowMs; return; }
    if (sPeerCount >= MAX_NODES) return;
    memcpy(sPeerMacAddresses[sPeerCount], peerMacAddress, 6);
    sPeerNodeIds[sPeerCount] = peerNodeId;
    sPeerLastSeenMs[sPeerCount] = nowMs;
    sPeerCount++;
}

static int NodeIdForMacAddress(const uint8_t *peerMacAddress) {
    for (int i = 0; i < sPeerCount; i++) if (memcmp(sPeerMacAddresses[i], peerMacAddress, 6) == 0) return sPeerNodeIds[i];
    return -1;  // not a mesh node, so not our sensing traffic
}

static int IndexForNodeId(int nodeId) {
    for (int i = 0; i < sPeerCount; i++) if (sPeerNodeIds[i] == nodeId) return i;
    return -1;
}

// --- Dynamic ring helpers (self is always alive; peers expire after LIVE_TIMEOUT_MS) ---
static int CountAlivePeers(void) {
    uint32_t nowMs = CurrentTimeMs(); int aliveCount = 0;
    for (int i = 0; i < sPeerCount; i++)
        if (sPeerNodeIds[i] != NODE_ID && nowMs - sPeerLastSeenMs[i] < LIVE_TIMEOUT_MS) aliveCount++;
    return aliveCount;
}
// Lowest live id, including self, is the leader.
static int FindLeaderNodeId(void) {
    uint32_t nowMs = CurrentTimeMs(); int leaderNodeId = NODE_ID;
    for (int i = 0; i < sPeerCount; i++)
        if (nowMs - sPeerLastSeenMs[i] < LIVE_TIMEOUT_MS && sPeerNodeIds[i] < leaderNodeId) leaderNodeId = sPeerNodeIds[i];
    return leaderNodeId;
}
// Next live id strictly greater than `afterNodeId`, else wraps to the lowest live id. Self is a candidate.
static int NextLiveNodeIdAfter(int afterNodeId) {
    uint32_t nowMs = CurrentTimeMs();
    int nextGreaterNodeId = 0x7fffffff, lowestLiveNodeId = NODE_ID;
    for (int i = 0; i < sPeerCount; i++) {
        if (nowMs - sPeerLastSeenMs[i] >= LIVE_TIMEOUT_MS) continue;
        if (sPeerMissedTurnCount[i] >= TOKEN_MISS_MAX) continue;  // adaptive skip: unresponsive to recent handoff
        int candidateNodeId = sPeerNodeIds[i];
        if (candidateNodeId > afterNodeId && candidateNodeId < nextGreaterNodeId) nextGreaterNodeId = candidateNodeId;
        if (candidateNodeId < lowestLiveNodeId) lowestLiveNodeId = candidateNodeId;
    }
    if (NODE_ID > afterNodeId && NODE_ID < nextGreaterNodeId) nextGreaterNodeId = NODE_ID;  // self
    return nextGreaterNodeId != 0x7fffffff ? nextGreaterNodeId : lowestLiveNodeId;
}

// Shared epoch-ms once SNTP has synced; monotonic esp_timer ms as a fallback before then.
static uint64_t WallClockMs(void) {
    struct timeval currentTime;
    gettimeofday(&currentTime, NULL);
    if (currentTime.tv_sec > 1600000000)  // plausible real time, so SNTP is synced
        return (uint64_t)currentTime.tv_sec * 1000 + currentTime.tv_usec / 1000;
    return (uint64_t)(esp_timer_get_time() / 1000);
}

// CsiReceiveCallback runs on Core 0 at ~300fps. No malloc/formatting allowed.
static void CsiReceiveCallback(void *context, wifi_csi_info_t *info) {
    if (!info || !info->buf) return;
    sCsiCallbackCount++;  // diag: count every CSI callback regardless of source
    int transmitterNodeId = NodeIdForMacAddress(info->mac);
    if (transmitterNodeId < 0) return;  // only emit CSI from known mesh nodes
#if WIFI_CSI_PHY_GAIN_ENABLE
    // Freeze AGC/FFT once a baseline is built. Room needs to stay still for the first few seconds.
    if (!sGainLockState) {
        uint8_t currentAgc; int8_t currentFft;
        esp_csi_gain_ctrl_get_rx_gain(&info->rx_ctrl, &currentAgc, &currentFft);
        if (sGainBaselineSampleCount < GAIN_BASELINE_PACKET_COUNT) {
            esp_csi_gain_ctrl_record_rx_gain(currentAgc, currentFft);
            sGainBaselineSampleCount++;
        } else {
            uint8_t baselineAgc; int8_t baselineFft;
            if (esp_csi_gain_ctrl_get_rx_gain_baseline(&baselineAgc, &baselineFft) == ESP_OK) {
                sLockedAgcValue = baselineAgc; sLockedFftValue = baselineFft;
                // Skip lock if AGC < 30 (signal too strong, forcing causes WDT crash). Rely on host CV normalization.
                if (baselineAgc < 30) {
                    sGainLockState = 2;  // SKIP: too strong
                } else {
                    esp_csi_gain_ctrl_set_rx_force_gain(baselineAgc, baselineFft);
                    sGainLockState = 1;  // LOCK
                }
            } else {
                sGainBaselineSampleCount = 0;  // baseline not ready yet, recollect
            }
        }
    }
#endif
    // Pin CSI width to keep (1,S) shape and collapse STBC frames instead of dropping them.
    int csiLengthBytes = info->len;
    if (csiLengthBytes <= 0 || csiLengthBytes > 2 * CSI_MAX_BYTES) return;
    if (sExpectedCsiLengthBytes == 0) {
        if (csiLengthBytes > CSI_MAX_BYTES) return;            // wait for a base-width frame to pin to
        sExpectedCsiLengthBytes = csiLengthBytes;
    }
    if (csiLengthBytes == 2 * sExpectedCsiLengthBytes) csiLengthBytes = sExpectedCsiLengthBytes;   // STBC double, keep first LTF block only
    else if (csiLengthBytes != sExpectedCsiLengthBytes) return;              // off-format (short/odd), drop like the host

    // Drop frames exceeding CSI_MAX_HZ to prevent UDP sendto ENOMEM overruns.
#if CSI_MAX_HZ > 0
    static int64_t sLastEmitTimeUs = 0;
    const int64_t nowUs = esp_timer_get_time();
    if (nowUs - sLastEmitTimeUs < 1000000 / CSI_MAX_HZ) return;
    sLastEmitTimeUs = nowUs;
#endif

    CsiRawFrame csiFrame;
    memcpy(csiFrame.mac, info->mac, 6);
    csiFrame.len = (uint16_t)csiLengthBytes;
    csiFrame.ts_us = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFF);
    memcpy(csiFrame.buf, info->buf, csiLengthBytes);
    if (xQueueSend(sCsiFrameQueue, &csiFrame, 0) == pdTRUE) sCsiFrameCount++;
}

static volatile uint32_t sLastTokenAcceptedMs = 0;  // dedup: when we last accepted a token...
static volatile uint8_t  sLastTokenTransmitterNodeId = 0;  // ...and from which tx node (token repeats TOKEN_REPEAT x)
static volatile int  sHandoffTargetNodeId = -1;        // node we last handed the token to (adaptive-skip pending)
static volatile bool sbHandoffHeard = false;  // did that node transmit in response to our handoff?

// Update liveness, accept token on burst handoff.
static void EspNowReceiveCallback(const esp_now_recv_info_t *info, const uint8_t *data, int packetLengthBytes) {
    if (packetLengthBytes < (int)sizeof(MeshPacket)) return;
    const MeshPacket *packet = (const MeshPacket *)data;
    if (packet->magic != MESH_MAGIC) return;
    sLastAirTimeMs = CurrentTimeMs();
    LearnPeerMacAddress(info->src_addr, packet->tx_id);
    if (packet->tx_id == sHandoffTargetNodeId) sbHandoffHeard = true;  // it took its turn, so it's responsive
    int peerIndex = IndexForNodeId(packet->tx_id);
    if (peerIndex >= 0) sPeerMissedTurnCount[peerIndex] = 0;                           // hearing it transmit clears its skip count
    // Token rides last TOKEN_REPEAT frames. Dedup within TOKEN_DEDUP_MS to prevent multi-bursts.
    if ((packet->flags & FLAG_LAST) && packet->next_id == NODE_ID) {
        uint32_t nowMs = CurrentTimeMs();
        if (!(packet->tx_id == sLastTokenTransmitterNodeId && nowMs - sLastTokenAcceptedMs < TOKEN_DEDUP_MS)) {
            sLastTokenTransmitterNodeId = packet->tx_id;
            sLastTokenAcceptedMs = nowMs;
            uint8_t tokenSignal = 1;
            xQueueSend(sTurnQueue, &tokenSignal, 0);  // it's our turn next
        }
    }
}

// Transmit one burst, handing the token to the next LIVE node id on the final frame.
static volatile uint32_t sLastTransmitMs = 0;

static void SendTokenBurst(void) {
    sLastTransmitMs = CurrentTimeMs();
    // Adaptive skip: drop unresponsive node after TOKEN_MISS_MAX misses instead of waiting for timeout.
    if (sHandoffTargetNodeId >= 0 && sHandoffTargetNodeId != NODE_ID && !sbHandoffHeard) {
        int peerIndex = IndexForNodeId(sHandoffTargetNodeId);
        if (peerIndex >= 0 && sPeerMissedTurnCount[peerIndex] < 255) sPeerMissedTurnCount[peerIndex]++;
    }
    uint8_t nextNodeId = (uint8_t)NextLiveNodeIdAfter(NODE_ID);
    sHandoffTargetNodeId = (nextNodeId != NODE_ID) ? nextNodeId : -1;
    sbHandoffHeard = false;
    for (int frameIndex = 0; frameIndex < BURST_LEN; frameIndex++) {
        // Repeat token on last TOKEN_REPEAT frames to survive drops.
        MeshPacket tokenFrame = {MESH_MAGIC, (uint8_t)NODE_ID, (uint8_t)frameIndex,
                        (uint8_t)(frameIndex >= BURST_LEN - TOKEN_REPEAT ? FLAG_LAST : 0), nextNodeId};
        // Retry token frames on TX NO_MEM to avoid ring stall.
        esp_err_t sendResult;
        for (int retryAttempt = 0; (sendResult = esp_now_send(BROADCAST_MAC_ADDRESS, (uint8_t *)&tokenFrame, sizeof(tokenFrame))) == ESP_ERR_ESPNOW_NO_MEM && retryAttempt < 3; retryAttempt++)
            vTaskDelay(1);
        if (sendResult == ESP_OK) sTxSentCount++;
        vTaskDelay(pdMS_TO_TICKS(BURST_MS));
    }
}

// Token loop: leader bootstraps/heals token. Nodes burst when handed turn.
static volatile bool sbEverGotTurn = false;  // admitted to the ring once we've received >=1 token

static void MeshTask(void *) {
    vTaskDelay(pdMS_TO_TICKS(3000));  // settle: let STA associate and hear peers before deciding the ring
    if (sbConnected && NODE_ID == FindLeaderNodeId()) {
        ESP_LOGI(sTag, "bootstrap: leader, starting mesh");
        SendTokenBurst();
    }
    for (;;) {
        uint8_t tokenSignal;
        if (xQueueReceive(sTurnQueue, &tokenSignal, pdMS_TO_TICKS(TURN_TIMEOUT_MS)) == pdTRUE) {
            sbEverGotTurn = true;  // we're in the ring
            if (sbConnected) SendTokenBurst();
        } else if (sbConnected) {
            uint32_t nowMs = CurrentTimeMs();
            if (NODE_ID == FindLeaderNodeId()) {
                if (nowMs - sLastAirTimeMs > TURN_TIMEOUT_MS) {
                    ESP_LOGD(sTag, "token lost, restarting");
                    SendTokenBurst();
                }
            } else if (nowMs - sLastAirTimeMs > LEADER_DEAD_MS + (uint32_t)NODE_ID * 100) {
                // Dead ring backstop: restart ring with id*100 stagger so lowest ID fires first.
                ESP_LOGD(sTag, "leader dead, taking over");
                SendTokenBurst();
            } else {
                // Send fast keep-alive until admitted, then slow to 5s.
                uint32_t keepAliveIntervalMs = sbEverGotTurn ? 5000 : DISCOVERY_MS;
                if (nowMs - sLastTransmitMs > keepAliveIntervalMs) {
                    MeshPacket keepAlivePacket = {MESH_MAGIC, (uint8_t)NODE_ID, 0, 0, 0};
                    esp_now_send(BROADCAST_MAC_ADDRESS, (uint8_t *)&keepAlivePacket, sizeof(keepAlivePacket));
                    sLastTransmitMs = nowMs;
                }
            }
        }
    }
}

// Prepend header and ship datagram.
static int SendCsiBatch(int socketFd, struct sockaddr_in *destinationAddress, uint8_t *datagramBuffer,
                          const uint8_t *csiRecordBytes, int bodyLengthBytes, int recordCount) {
    CsiBatchHeader header = {MESH_MAGIC, 2, (uint8_t)NODE_ID, WallClockMs(), (uint16_t)recordCount};
    memcpy(datagramBuffer, &header, sizeof(header));
    memcpy(datagramBuffer + sizeof(header), csiRecordBytes, bodyLengthBytes);
    destinationAddress->sin_addr = sPcAddress;
    return sendto(socketFd, datagramBuffer, sizeof(header) + bodyLengthBytes, 0, (struct sockaddr *)destinationAddress, sizeof(*destinationAddress));
}

// Flush batch. Retry transient ENOMEM to prevent queue overflow.
static bool FlushCsiBatch(int socketFd, struct sockaddr_in *destinationAddress, uint8_t *datagramBuffer,
                            const uint8_t *csiRecordBytes, int bodyLengthBytes, int recordCount) {
    for (int retryAttempt = 0; retryAttempt < UDP_SEND_RETRIES; retryAttempt++) {
        if (SendCsiBatch(socketFd, destinationAddress, datagramBuffer, csiRecordBytes, bodyLengthBytes, recordCount) >= 0) return true;
        if (errno != ENOMEM) return true;       // not a buffer issue, so retrying won't help
        vTaskDelay(pdMS_TO_TICKS(2));           // let the Wi-Fi TX path drain, then resend the same batch
    }
    return false;
}

// Pack CSI into batch datagrams.
static void UdpBatchTask(void *) {
    int socketFd = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in destinationAddress = {};
    destinationAddress.sin_family = AF_INET;
    destinationAddress.sin_port = htons(CSI_UDP_PORT);
    static uint8_t csiRecordBytes[UDP_BATCH_BYTES];
    static uint8_t datagramBuffer[sizeof(CsiBatchHeader) + UDP_BATCH_BYTES];
    int bodyLengthBytes = 0, recordCount = 0, framesSinceYield = 0;
    CsiRawFrame csiFrame;
    int64_t backoffUntilUs = 0;
    for (;;) {
        int64_t nowUs = esp_timer_get_time();
        if (backoffUntilUs > 0) {
            if (nowUs < backoffUntilUs) {
                vTaskDelay(pdMS_TO_TICKS(5));
                continue;
            } else {
                ESP_LOGI(sTag, "ENOMEM backoff done, resuming sends");
                backoffUntilUs = 0;
            }
        }

        if (xQueueReceive(sCsiFrameQueue, &csiFrame, pdMS_TO_TICKS(BATCH_MS)) == pdTRUE) {
            int recordSizeBytes = 6 + 4 + 2 + csiFrame.len;   // mac | ts_us | len | raw int8 CSI
            if (bodyLengthBytes + recordSizeBytes > UDP_BATCH_BYTES && recordCount > 0) {
                if (!FlushCsiBatch(socketFd, &destinationAddress, datagramBuffer, csiRecordBytes, bodyLengthBytes, recordCount)) {
                    backoffUntilUs = esp_timer_get_time() + UDP_ENOMEM_BACKOFF_MS * 1000;
                    ESP_LOGW(sTag, "sendto ENOMEM after %d retries, backing off %d ms",
                             UDP_SEND_RETRIES, UDP_ENOMEM_BACKOFF_MS);
                }
                bodyLengthBytes = 0; recordCount = 0;   // batch left the air (or dropped after retries), so reset
            }
            if (recordSizeBytes <= UDP_BATCH_BYTES) {
                uint8_t *writePointer = csiRecordBytes + bodyLengthBytes;
                memcpy(writePointer, csiFrame.mac, 6);          writePointer += 6;
                memcpy(writePointer, &csiFrame.ts_us, 4);        writePointer += 4;   // LE u32
                uint16_t recordLengthBytes = csiFrame.len;
                memcpy(writePointer, &recordLengthBytes, 2);              writePointer += 2;   // LE u16
                memcpy(writePointer, csiFrame.buf, csiFrame.len);
                bodyLengthBytes += recordSizeBytes; recordCount++;
            }
            // Yield ~1ms every 64 frames to prevent Core-1 starvation if queue is full.
            if (++framesSinceYield >= 64) { framesSinceYield = 0; vTaskDelay(1); }
        } else if (recordCount > 0) {
            if (!FlushCsiBatch(socketFd, &destinationAddress, datagramBuffer, csiRecordBytes, bodyLengthBytes, recordCount)) {
                backoffUntilUs = esp_timer_get_time() + UDP_ENOMEM_BACKOFF_MS * 1000;
                ESP_LOGW(sTag, "sendto ENOMEM after %d retries, backing off %d ms",
                         UDP_SEND_RETRIES, UDP_ENOMEM_BACKOFF_MS);
            }
            bodyLengthBytes = 0; recordCount = 0;
        }
    }
}

// PC heartbeat via HEALTH_UDP_PORT.
static void HealthTask(void *) {
    int socketFd = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in destinationAddress = {};
    destinationAddress.sin_family = AF_INET;
    destinationAddress.sin_port = htons(HEALTH_UDP_PORT);
    char healthMessage[288];
    uint32_t lastCsiCount = 0, lastTxSentCount = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(HEALTH_MS));
        wifi_ap_record_t apRecord; int rssiDbm = 0;
        if (esp_wifi_sta_get_ap_info(&apRecord) == ESP_OK) rssiDbm = apRecord.rssi;
        uint32_t csiCount = sCsiFrameCount, txSentCount = sTxSentCount;
        const char *gainStateLabel = sGainLockState == 1 ? "LOCK" : sGainLockState == 2 ? "SKIP" : "coll";
        int messageLength = snprintf(healthMessage, sizeof(healthMessage),
            "{\"v\":1,\"node\":%d,\"type\":\"health\",\"up_s\":%llu,\"heap\":%lu,\"rssi\":%d,"
            "\"csi_hz\":%lu,\"tx_hz\":%lu,\"peers\":%d,\"leader\":%d,\"gain\":\"%s\",\"agc\":%u,"
            "\"synced\":%d}\n",
            NODE_ID, (unsigned long long)(esp_timer_get_time() / 1000000),
            (unsigned long)esp_get_free_heap_size(), rssiDbm,
            (unsigned long)((csiCount - lastCsiCount) * 1000 / HEALTH_MS),
            (unsigned long)((txSentCount - lastTxSentCount) * 1000 / HEALTH_MS),
            CountAlivePeers(), FindLeaderNodeId(), gainStateLabel, (unsigned)sLockedAgcValue, WallClockMs() > 1600000000000ULL ? 1 : 0);
        destinationAddress.sin_addr = sPcAddress;
        int sendResult = sendto(socketFd, healthMessage, messageLength, 0, (struct sockaddr *)&destinationAddress, sizeof(destinationAddress));
        if (sendResult < 0) {
            ESP_LOGW(sTag, "health sendto failed: errno %d", errno);
        }
        lastCsiCount = csiCount; lastTxSentCount = txSentCount;
    }
}

// Serial heartbeat for USB monitor.
static void StatsTask(void *) {
    uint32_t lastCsiCount = 0, lastCsiRawCount = 0, lastTxSentCount = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        uint32_t csiCount = sCsiFrameCount, csiRawCount = sCsiCallbackCount, txSentCount = sTxSentCount;
        wifi_ap_record_t apRecord; int rssiDbm = 0;
        if (esp_wifi_sta_get_ap_info(&apRecord) == ESP_OK) rssiDbm = apRecord.rssi;
        ESP_LOGI(sTag,
            "node=%d csi_hz=%lu csi_raw=%lu tx=%lu peers=%d leader=%d gain=%s(%u/%d) rssi=%d heap=%lu",
            NODE_ID, (unsigned long)(csiCount - lastCsiCount), (unsigned long)(csiRawCount - lastCsiRawCount),
            (unsigned long)(txSentCount - lastTxSentCount), CountAlivePeers(), FindLeaderNodeId(),
            sGainLockState == 1 ? "LOCK" : sGainLockState == 2 ? "SKIP" : "coll",
            (unsigned)sLockedAgcValue, (int)sLockedFftValue, rssiDbm, (unsigned long)esp_get_free_heap_size());
        lastCsiCount = csiCount; lastCsiRawCount = csiRawCount; lastTxSentCount = txSentCount;
    }
}

#if STATUS_LED_GPIO >= 0
// WS2812 status LED. Do not define RGB_PWR_GPIO or it kills the LED.
static void LedTask(void *) {
#ifdef RGB_PWR_GPIO
    esp_rom_gpio_pad_select_gpio(RGB_PWR_GPIO);
    gpio_set_direction((gpio_num_t)RGB_PWR_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level((gpio_num_t)RGB_PWR_GPIO, 1);
#endif
    led_strip_handle_t ledStrip = NULL;
    led_strip_config_t stripConfig = {
        .strip_gpio_num = STATUS_LED_GPIO,
        .max_leds = 1,
        .led_pixel_format = LED_PIXEL_FORMAT_GRB,
        .led_model = LED_MODEL_WS2812,
        .flags = { .invert_out = false },
    };
    led_strip_rmt_config_t rmtConfig = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10 * 1000 * 1000,
        .flags = { .with_dma = false },
    };
    if (led_strip_new_rmt_device(&stripConfig, &rmtConfig, &ledStrip) != ESP_OK) vTaskDelete(NULL);

    uint32_t lastCsiCount = 0;
    for (;;) {
        uint32_t csiCount = sCsiFrameCount, csiDelta = csiCount - lastCsiCount; lastCsiCount = csiCount;
        int aliveCount = CountAlivePeers();
        uint8_t red = 0, green = 0, blue = 0, brightness = 32;
        if (!sbConnected)                  { red = brightness; }            // red: no Wi-Fi
        else if (sGainLockState == 0)      { red = brightness; green = brightness/2; }  // yellow: calibrating
        else if (aliveCount == 0)          { blue = brightness; }            // blue: solo
        else if (csiDelta == 0)            { red = brightness; blue = brightness; }    // magenta: silent mesh
        else                                { green = brightness; }            // green: healthy
        led_strip_set_pixel(ledStrip, 0, red, green, blue);
        led_strip_refresh(ledStrip);
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}
#endif

// Promiscuous mode is enabled only so the CSI engine taps non-AP frames; the packets go unused.
static void PromiscuousReceiveCallback(void *packetBuffer, wifi_promiscuous_pkt_type_t packetType) {}

static void WifiEventHandler(void *, esp_event_base_t eventBase, int32_t eventId, void *eventData) {
    if (eventBase == WIFI_EVENT && eventId == WIFI_EVENT_STA_DISCONNECTED) {
        sbConnected = false;   // stop bursting so the radio is free to re-associate fast
        esp_wifi_connect();
    } else if (eventBase == IP_EVENT && eventId == IP_EVENT_STA_GOT_IP) {
        sbConnected = true;
        ip_event_got_ip_t *gotIpEvent = (ip_event_got_ip_t *)eventData;
        ESP_LOGI(sTag, "router ip: " IPSTR, IP2STR(&gotIpEvent->ip_info.ip));
        // Set HT rate post-association since 40MHz secondary channel only exists then.
        esp_now_rate_config_t rateConfig = {
            .phymode = WT_BW_HT40 ? WIFI_PHY_MODE_HT40 : WIFI_PHY_MODE_HT20,
            .rate = WIFI_PHY_RATE_MCS0_LGI,
            .ersu = false,
            .dcm = false,
        };
        esp_err_t rateConfigResult = esp_now_set_peer_rate_config(BROADCAST_MAC_ADDRESS, &rateConfig);
        ESP_LOGI(sTag, "esp_now HT%d rate: %s", WT_BW_HT40 ? 40 : 20, esp_err_to_name(rateConfigResult));

        // Warning if router is 20MHz-only while WT_BW_HT40=1.
        wifi_ap_record_t apRecord = {};
        wifi_bandwidth_t bandwidth = WIFI_BW_HT20;
        esp_wifi_get_bandwidth(WIFI_IF_STA, &bandwidth);
        if (esp_wifi_sta_get_ap_info(&apRecord) == ESP_OK) {
            const char *secondaryChannelLabel = apRecord.second == WIFI_SECOND_CHAN_ABOVE ? "ABOVE(=40MHz)"
                            : apRecord.second == WIFI_SECOND_CHAN_BELOW ? "BELOW(=40MHz)"
                            : "NONE(=20MHz only)";
            ESP_LOGW(sTag, "AP-CHECK ssid=%s ch=%d second=%s 11n=%d rssi=%d sta_bw=%s", apRecord.ssid,
                     apRecord.primary, secondaryChannelLabel, apRecord.phy_11n, apRecord.rssi, bandwidth == WIFI_BW_HT40 ? "HT40" : "HT20");
        }
    }
}

extern "C" void app_main(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_t *staNetif = esp_netif_create_default_wifi_sta();
    char hostname[20];
    snprintf(hostname, sizeof(hostname), "wavetrace%d", NODE_ID);
    esp_netif_set_hostname(staNetif, hostname);

    wifi_init_config_t wifiInitConfig = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&wifiInitConfig);
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, WifiEventHandler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, WifiEventHandler, NULL);
    esp_wifi_set_mode(WIFI_MODE_STA);
    wifi_config_t staConfig = {};
    strncpy((char *)staConfig.sta.ssid, ROUTER_SSID, sizeof(staConfig.sta.ssid));
    strncpy((char *)staConfig.sta.password, ROUTER_PASS, sizeof(staConfig.sta.password));
    esp_wifi_set_config(WIFI_IF_STA, &staConfig);
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
    esp_sntp_config_t sntpConfig = ESP_NETIF_SNTP_DEFAULT_CONFIG(SNTP_SERVER);
    esp_netif_sntp_init(&sntpConfig);

    // ESP-NOW on the STA interface uses the current (router) channel, so every node hears every other.
    esp_now_init();
    esp_now_register_recv_cb(EspNowReceiveCallback);
    esp_now_peer_info_t broadcastPeer = {};
    memcpy(broadcastPeer.peer_addr, BROADCAST_MAC_ADDRESS, 6);
    broadcastPeer.channel = 0;            // 0 = current channel (locked by the STA association)
    broadcastPeer.ifidx = WIFI_IF_STA;
    broadcastPeer.encrypt = false;
    esp_now_add_peer(&broadcastPeer);

    // Use HT rate for sensing frames to trigger CSI reliably.

    // Promiscuous mode allows tapping peer ESP-NOW bursts. Filter DATA frames to prevent RX buffer crashes.
    esp_wifi_set_promiscuous_rx_cb(PromiscuousReceiveCallback);
    wifi_promiscuous_filter_t promiscuousFilter = {
        .filter_mask = WIFI_PROMIS_FILTER_MASK_MGMT,
    };
    esp_wifi_set_promiscuous_filter(&promiscuousFilter);
    esp_wifi_set_promiscuous(true);

    wifi_csi_config_t csiConfig = {
        .lltf_en = true, .htltf_en = true, .stbc_htltf2_en = true,
        .ltf_merge_en = true, .channel_filter_en = false, .manu_scale = false,
    };
    esp_wifi_set_csi_config(&csiConfig);
    esp_wifi_set_csi_rx_cb(CsiReceiveCallback, NULL);
    esp_wifi_set_csi(true);

    // 128-frame by-value queue (~50KB) prevents TX heap starvation.
    sCsiFrameQueue = xQueueCreate(128, sizeof(CsiRawFrame));
    sTurnQueue = xQueueCreate(4, sizeof(uint8_t));
    // Pin every app task to Core 1 (APP_CPU), leaving Core 0 (PRO_CPU) dedicated to the Wi-Fi/lwIP stack and CsiReceiveCallback.
    xTaskCreatePinnedToCore(DiscoveryTask, "discovery", 3072, NULL, 4, NULL, 1);
    xTaskCreatePinnedToCore(UdpBatchTask, "csi_udp", 4096, NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(MeshTask, "mesh", 4096, NULL, 6, NULL, 1);
    xTaskCreatePinnedToCore(HealthTask, "health", 4096, NULL, 3, NULL, 1);
    xTaskCreatePinnedToCore(StatsTask, "stats", 4096, NULL, 2, NULL, 1);
#if STATUS_LED_GPIO >= 0
    xTaskCreatePinnedToCore(LedTask, "led", 3072, NULL, 1, NULL, 1);
#endif
}
