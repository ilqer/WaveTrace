idf_component_register(SRCS "main.cpp" INCLUDE_DIRS ".")

# flash.sh injects the per-board NODE_ID via the environment (integer — no string quoting).
# Falls back to the config.h default when NODE_ID is unset.
if(DEFINED ENV{NODE_ID})
  add_compile_definitions(NODE_ID=$ENV{NODE_ID})
endif()
