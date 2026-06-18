// ESP32-S3 dedicated TX = STA that joins the RX's SoftAP and floods UDP packets (ESP32-CSI-Tool
// "active_sta"). Each packet is an 802.11 data frame the RX captures CSI from. Placeable illuminator.
// Build with ESP-IDF v5.x (esp32s3). Bring up the RX (SoftAP) first, then this board joins it.

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

static const char *TAG = "wt_tx";

// Auto-reconnect when the RX SoftAP reboots (e.g. after reflash).
static void wifi_event_handler(void *, esp_event_base_t base, int32_t id, void *) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGI(TAG, "disconnected — reconnecting");
        esp_wifi_connect();
    }
}

extern "C" void app_main(void) {
    nvs_flash_init();
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event_handler, NULL);
    esp_wifi_set_mode(WIFI_MODE_STA);
    wifi_config_t wcfg = {};
    strncpy((char *)wcfg.sta.ssid, AP_SSID, sizeof(wcfg.sta.ssid));
    strncpy((char *)wcfg.sta.password, AP_PASS, sizeof(wcfg.sta.password));
    esp_wifi_set_config(WIFI_IF_STA, &wcfg);
    esp_wifi_start();
    esp_wifi_set_ps(WIFI_PS_NONE);  // no modem-sleep -> steady 100 Hz flood
    esp_wifi_connect();

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    struct sockaddr_in dest;
    memset(&dest, 0, sizeof(dest));
    dest.sin_family = AF_INET;
    dest.sin_port = htons(DEST_PORT);
    inet_pton(AF_INET, DEST_IP, &dest.sin_addr);

    static uint8_t payload[100] = {0};
    const TickType_t delay = pdMS_TO_TICKS(1000 / FLOOD_HZ);
    uint32_t sent = 0, errs = 0;
    for (;;) {
        int r = sendto(sock, payload, sizeof(payload), 0, (struct sockaddr *)&dest, sizeof(dest));
        if (r < 0) errs++; else sent++;
        if ((sent + errs) % FLOOD_HZ == 0) {  // ~1/sec; errs climb until associated, then sent climbs
            uint8_t prim = 0; wifi_second_chan_t sec;
            esp_wifi_get_channel(&prim, &sec);
            ESP_LOGI(TAG, "ch=%u sent=%lu errs=%lu", prim, (unsigned long)sent, (unsigned long)errs);
        }
        vTaskDelay(delay);
    }
}
