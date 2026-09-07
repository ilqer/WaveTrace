// ESP32-S3 dedicated TX. STA joins RX's SoftAP and floods UDP packets.
// RX captures CSI from these packets. Bring RX up first.

#include <stdio.h>
#include <string.h>
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "config.h"

static const char *sTag = "wt_tx";

// Reconnects when RX SoftAP reboots.
static void WifiEventHandler(void *, esp_event_base_t eventBase, int32_t eventId, void *) {
    if (eventBase == WIFI_EVENT && eventId == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(sTag, "disconnected, reconnecting");
        esp_wifi_connect();
    }
}

extern "C" void app_main(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t wifiInitConfig = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&wifiInitConfig);
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, WifiEventHandler, NULL);
    esp_wifi_set_mode(WIFI_MODE_STA);
    wifi_config_t staConfig = {};
    strncpy((char *)staConfig.sta.ssid, AP_SSID, sizeof(staConfig.sta.ssid));
    strncpy((char *)staConfig.sta.password, AP_PASS, sizeof(staConfig.sta.password));
    esp_wifi_set_config(WIFI_IF_STA, &staConfig);
    esp_wifi_start();
    esp_wifi_set_ps(WIFI_PS_NONE);  // Disable modem-sleep for steady flood.
    esp_wifi_connect();

    int floodSocket = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in destinationAddress;
    memset(&destinationAddress, 0, sizeof(destinationAddress));
    destinationAddress.sin_family = AF_INET;
    destinationAddress.sin_port = htons(DEST_PORT);
    inet_pton(AF_INET, DEST_IP, &destinationAddress.sin_addr);

    static uint8_t payload[100] = {0};
    const TickType_t floodDelayTicks = pdMS_TO_TICKS(1000 / FLOOD_HZ);
    uint32_t sentCount = 0, errorCount = 0;
    for (;;) {
        int sendResult = sendto(floodSocket, payload, sizeof(payload), 0, (struct sockaddr *)&destinationAddress, sizeof(destinationAddress));
        if (sendResult < 0) errorCount++; else sentCount++;
        if ((sentCount + errorCount) % FLOOD_HZ == 0) {  // ~once/sec print stats.
            uint8_t primaryChannel = 0; wifi_second_chan_t secondaryChannel;
            esp_wifi_get_channel(&primaryChannel, &secondaryChannel);
            ESP_LOGI(sTag, "ch=%u sent=%lu errs=%lu", primaryChannel, (unsigned long)sentCount, (unsigned long)errorCount);
        }
        vTaskDelay(floodDelayTicks);
    }
}
