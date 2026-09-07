// ESP32-S3 CSI receiver (APSTA). SoftAP captures CSI from TX UDP flood. STA joins router to unicast CSI to PC_IP.
// SoftAP channel matches router.

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

static const char *sTag = "wt_rx";

static char sJsonScratchBuffer[384 * 8 + 8];  // scratch buffer for the JSON int array; CsiReceiveCallback is single-threaded
static QueueHandle_t sCsiPacketQueue;        // queue of malloc'd (char *) UDP parse_batch packets
static volatile uint32_t sCsiFrameCount = 0;

static void CsiReceiveCallback(void *context, wifi_csi_info_t *info) {
    if (!info || !info->buf) return;
    int8_t *buf = info->buf;
    int subcarrierCount = info->len / 2;
    if (subcarrierCount <= 0 || subcarrierCount > 384) return;
    char macAddressString[18];
    snprintf(macAddressString, sizeof(macAddressString), "%02x:%02x:%02x:%02x:%02x:%02x",
             info->mac[0], info->mac[1], info->mac[2],
             info->mac[3], info->mac[4], info->mac[5]);
    int dataBufferPosition = 0;
    sJsonScratchBuffer[dataBufferPosition++] = '[';
    for (int k = 0; k < subcarrierCount; k++) {
        dataBufferPosition += snprintf(sJsonScratchBuffer + dataBufferPosition, sizeof(sJsonScratchBuffer) - dataBufferPosition,
                         k ? ",%d,%d" : "%d,%d", buf[2 * k], buf[2 * k + 1]);
    }
    sJsonScratchBuffer[dataBufferPosition++] = ']'; sJsonScratchBuffer[dataBufferPosition] = '\0';
    uint32_t timestampUs = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFF);

    // build the parse_batch packet and hand it to the UDP sender task
    size_t packetCapacityBytes = (size_t)dataBufferPosition + 200;
    char *packet = (char *)malloc(packetCapacityBytes);
    if (!packet) return;
    int headerLength = snprintf(packet, packetCapacityBytes, "{\"v\":1,\"node\":%d,\"ntp_ms\":%lu,\"n\":1}\n",
                       NODE_ID, (unsigned long)(timestampUs / 1000));
    snprintf(packet + headerLength, packetCapacityBytes - headerLength,
             "CSI_DATA,0,%s,0,0,0,0,0,0,0,0,0,0,0,0,0,%d,0,%u,0,0,0,%d,0,%s\n",
             macAddressString, AP_CHANNEL, (unsigned)timestampUs, subcarrierCount * 2, sJsonScratchBuffer);
    if (xQueueSend(sCsiPacketQueue, &packet, 0) != pdTRUE) {
        free(packet);
    }

    sCsiFrameCount++;
}

// 1-second heartbeat.
static void StatsTask(void *) {
    uint32_t lastCsiCount = 0;
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(1000));
        uint32_t csiCount = sCsiFrameCount;
        ESP_LOGI(sTag, "csi_hz=%lu udp->%s:%d", (unsigned long)(csiCount - lastCsiCount), PC_IP, CSI_UDP_PORT);
        lastCsiCount = csiCount;
    }
}

static void UdpSenderTask(void *) {
    int udpSocket = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in destinationAddress = {};
    destinationAddress.sin_family = AF_INET;
    destinationAddress.sin_port = htons(CSI_UDP_PORT);
    inet_pton(AF_INET, PC_IP, &destinationAddress.sin_addr);
    char *packet;
    for (;;) {
        if (xQueueReceive(sCsiPacketQueue, &packet, portMAX_DELAY) == pdTRUE) {
            sendto(udpSocket, packet, strlen(packet), 0, (struct sockaddr *)&destinationAddress, sizeof(destinationAddress));
            free(packet);
        }
    }
}

static void WifiEventHandler(void *, esp_event_base_t eventBase, int32_t eventId, void *eventData) {
    if (eventBase == WIFI_EVENT && eventId == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(sTag, "router disconnected, reconnecting");
        esp_wifi_connect();
    } else if (eventBase == IP_EVENT && eventId == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *gotIpEvent = (ip_event_got_ip_t *)eventData;
        ESP_LOGI(sTag, "router ip: " IPSTR, IP2STR(&gotIpEvent->ip_info.ip));
    }
}

extern "C" void app_main(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_ap();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t wifiInitConfig = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&wifiInitConfig);
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, WifiEventHandler, NULL);
    esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, WifiEventHandler, NULL);

    // APSTA: SoftAP for TX, STA for router backhaul.
    esp_wifi_set_mode(WIFI_MODE_APSTA);

    wifi_config_t apConfig = {};
    strncpy((char *)apConfig.ap.ssid, AP_SSID, sizeof(apConfig.ap.ssid));
    apConfig.ap.ssid_len = strlen(AP_SSID);
    strncpy((char *)apConfig.ap.password, AP_PASS, sizeof(apConfig.ap.password));
    apConfig.ap.channel = AP_CHANNEL;
    apConfig.ap.max_connection = 4;
    apConfig.ap.authmode = WIFI_AUTH_WPA2_PSK;
    esp_wifi_set_config(WIFI_IF_AP, &apConfig);

    wifi_config_t staConfig = {};
    strncpy((char *)staConfig.sta.ssid, ROUTER_SSID, sizeof(staConfig.sta.ssid));
    strncpy((char *)staConfig.sta.password, ROUTER_PASS, sizeof(staConfig.sta.password));
    esp_wifi_set_config(WIFI_IF_STA, &staConfig);

    esp_wifi_start();
    esp_wifi_connect();

    // CSI capture configuration.
    wifi_csi_config_t csiConfig = {
        .lltf_en = true, .htltf_en = true, .stbc_htltf2_en = true,
        .ltf_merge_en = true, .channel_filter_en = false, .manu_scale = false,
    };
    esp_wifi_set_csi_config(&csiConfig);
    esp_wifi_set_csi_rx_cb(CsiReceiveCallback, NULL);
    esp_wifi_set_csi(true);

    sCsiPacketQueue = xQueueCreate(20, sizeof(char *));
    xTaskCreate(UdpSenderTask, "csi_udp", 4096, NULL, 5, NULL);
    xTaskCreate(StatsTask, "stats", 4096, NULL, 2, NULL);

    // Drain TX UDP flood to prevent lwIP buffer overflow.
    int floodDrainSocket = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in bindAddress = {};
    bindAddress.sin_family = AF_INET; bindAddress.sin_addr.s_addr = INADDR_ANY; bindAddress.sin_port = htons(UDP_PORT);
    bind(floodDrainSocket, (struct sockaddr *)&bindAddress, sizeof(bindAddress));
    char discardBuffer[128];
    for (;;) {
        recvfrom(floodDrainSocket, discardBuffer, sizeof(discardBuffer), 0, NULL, NULL);
    }
}
