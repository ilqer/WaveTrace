idf_component_register(SRCS "main.cpp" INCLUDE_DIRS "."
                       REQUIRES esp_wifi esp_event esp_netif nvs_flash esp_timer lwip mdns esp_driver_gpio)

# flash.sh injects the per-board NODE_ID as a CMake cache var (-DNODE_ID=N). A cache var is
# re-evaluated when its value changes, forcing a reconfigure + recompile — an ENV var is NOT
# re-read by CMake, so flashing board 2 from the same build/ dir would silently reuse board 1's id.
# Falls back to the config.h default (1) when NODE_ID is unset.
if(DEFINED NODE_ID)
  add_compile_definitions(NODE_ID=${NODE_ID})
endif()
