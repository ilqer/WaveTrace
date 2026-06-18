// ESP32-S3 CSI receiver — APSTA mode:
//   SoftAP "WaveTrace-RX1": TX joins and floods UDP; CSI is captured from those data frames.
//   STA: joins the home router (ROUTER_SSID) so the RX has a real IP and can send CSI unicast
//        to the PC (PC_IP) over the same LAN. PC stays on the router — no need to join
//        WaveTrace-RX1. macOS subnet-broadcast blocking is bypassed entirely.
//
// Channel note: in APSTA mode the SoftAP channel is forced to match the router's channel
// after STA connects. TX discovers the SoftAP on whatever channel it lands on via a scan.
// Build with ESP-IDF v5.x (esp32s3). Console baud 921600 (see sdkconfig).

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "lwip/sockets.h"
#include "config.h"

#ifndef NODE_ID
#define NODE_ID 1
#endif

static const char *TAG = "wt_rx";

static char data_buf[384 * 8 + 8];  // JSON int-array scratch (single-threaded csi_cb)
static QueueHandle_t s_csi_q;        // queue of malloc'd (char *) UDP parse_batch packets
static volatile uint32_t s_csi_count = 0;

static void csi_cb(void *ctx, wifi_csi_info_t *info) {
    if (!info || !info->buf) return;
    int8_t *buf = info->buf;
    int S = info->len / 2;
    if (S <= 0 || S > 384) return;
    char mac[18];
    snprintf(mac, sizeof(mac), "%02x:%02x:%02x:%02x:%02x:%02x",
             info->mac[0], info->mac[1], info->mac[2],
             info->mac[3], info->mac[4], info->mac[5]);
    int dpos = 0;
    data_buf[dpos++] = '[';
    for (int k = 0; k < S; k++) {
        dpos += snprintf(data_buf + dpos, sizeof(data_buf) - dpos,
                         k ? ",%d,%d" : "%d,%d", buf[2 * k], buf[2 * k + 1]);
    }
    data_buf[dpos++] = ']'; data_buf[dpos] = '\0';
    uint32_t ts_us = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFF);

    // Build parse_batch packet and enqueue for UDP sender task
    size_t cap = (size_t)dpos + 200;
    char *pkt = (char *)malloc(cap);
    if (!pkt) return;
    int hdr = snprintf(pkt, cap, "{\"v\":1,\"node\":%d,\"ntp_ms\":%lu,\"n\":1}\n",
                       NODE_ID, (unsigned long)(ts_us / 1000));
    snprintf(pkt + hdr, cap - hdr,
             "CSI_DATA,0,%s,0,0,0,0,0,0,0,0,0,0,0,0,0,%d,0,%u,0,0,0,%d,0,%s\n",
             mac, AP_CHANNEL, (unsigned)ts_us, S * 2, data_buf);
    if (xQueueSend(s_csi_q, &pkt, 0) != pdTRUE) {
        free(pkt);
    }

    s_csi_count++;
}

// 1-second heartbeat: safe to printf here because this is NOT the Wi-Fi task.
static void stats_task(void *) {
    uint32_t last = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        uint32_t n = s_csi_count;
        ESP_LOGI(TAG, "csi_hz=%lu udp->%s:%d", (unsigned long)(n - last), PC_IP, CSI_UDP_PORT);
        last = n;
    }
}

// Unicast CSI packets to PC_IP:CSI_UDP_PORT over the router (STA) interface.
static void udp_sender_task(void *) {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in dst = {};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(CSI_UDP_PORT);
    inet_pton(AF_INET, PC_IP, &dst.sin_addr);
    char *pkt;
    for (;;) {
        if (xQueueReceive(s_csi_q, &pkt, portMAX_DELAY) == pdTRUE) {
            sendto(sock, pkt, strlen(pkt), 0, (struct sockaddr *)&dst, sizeof(dst));
            free(pkt);
        }
    }
}

static void wifi_event_handler(void *, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "router disconnected — reconnecting");
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "router ip: " IPSTR, IP2STR(&ev->ip_info.ip));
    }
}

extern "C" void app_main(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_ap();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL);

    // APSTA: SoftAP for TX + STA for router backhaul
    esp_wifi_set_mode(WIFI_MODE_APSTA);

    wifi_config_t ap_cfg = {};
    strncpy((char *)ap_cfg.ap.ssid, AP_SSID, sizeof(ap_cfg.ap.ssid));
    ap_cfg.ap.ssid_len = strlen(AP_SSID);
    strncpy((char *)ap_cfg.ap.password, AP_PASS, sizeof(ap_cfg.ap.password));
    ap_cfg.ap.channel = AP_CHANNEL;
    ap_cfg.ap.max_connection = 4;
    ap_cfg.ap.authmode = WIFI_AUTH_WPA2_PSK;
    esp_wifi_set_config(WIFI_IF_AP, &ap_cfg);

    wifi_config_t sta_cfg = {};
    strncpy((char *)sta_cfg.sta.ssid, ROUTER_SSID, sizeof(sta_cfg.sta.ssid));
    strncpy((char *)sta_cfg.sta.password, ROUTER_PASS, sizeof(sta_cfg.sta.password));
    esp_wifi_set_config(WIFI_IF_STA, &sta_cfg);

    esp_wifi_start();
    esp_wifi_connect();  // STA connects to router

    // CSI capture (all preamble fields on; no channel filter/scale)
    wifi_csi_config_t csi_cfg = {
        .lltf_en = true, .htltf_en = true, .stbc_htltf2_en = true,
        .ltf_merge_en = true, .channel_filter_en = false, .manu_scale = false,
    };
    esp_wifi_set_csi_config(&csi_cfg);
    esp_wifi_set_csi_rx_cb(csi_cb, NULL);
    esp_wifi_set_csi(true);

    s_csi_q = xQueueCreate(20, sizeof(char *));
    xTaskCreate(udp_sender_task, "csi_udp", 4096, NULL, 5, NULL);
    xTaskCreate(stats_task, "stats", 4096, NULL, 2, NULL);

    // Drain TX's UDP flood on port UDP_PORT (prevents LWIP buffer exhaustion)
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in a = {};
    a.sin_family = AF_INET; a.sin_addr.s_addr = INADDR_ANY; a.sin_port = htons(UDP_PORT);
    bind(s, (struct sockaddr *)&a, sizeof(a));
    char rxbuf[128];
    for (;;) {
        recvfrom(s, rxbuf, sizeof(rxbuf), 0, NULL, NULL);
    }
}
