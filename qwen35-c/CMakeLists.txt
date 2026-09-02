cmake_minimum_required(VERSION 3.20)
project(qwen35_c C)

set(CMAKE_C_STANDARD 99)
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  add_compile_options(-mavx2 -mfma)
  # OpenMP: clang-cl uses -openmp (not /openmp:llvm which is silently
  # ignored). libomp must be linked explicitly — the flag alone only emits
  # the pragma hooks, it doesn't pull the runtime into the link line.
  if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    add_compile_options(-openmp)
    add_link_options("/LIBPATH:C:\\PROGRA~1\\LLVM\\lib" "libomp.lib")
    # Link-time code generation for cross-TU inlining
    add_compile_options(-flto)
    add_link_options(-flto)
  else()
    find_package(OpenMP REQUIRED)
    link_libraries(OpenMP::OpenMP_C)
    add_compile_options(-flto)
    add_link_options(-flto)
  endif()
endif()
if(MSVC AND NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
  add_compile_options(/arch:AVX2)
  add_compile_definitions(Q35_ENABLE_AVX2=1)
endif()

add_library(q35 STATIC
  src/q35_tok.c
  src/q35_plat_win.c
  src/q35_bpe.c
  src/q35_json.c
  src/q35_st.c
  src/q35_cfg.c
  src/q35_mm.c
  src/q35_kern.c
  src/q35_deltanet.c
  src/q35_attn.c
  src/q35_model.c
)
target_include_directories(q35 PUBLIC include PRIVATE src)
# PAL lives in the lib (q35_plat_win.c): winsock (TCP shim) and psapi
# (peak RSS) are OS deps of the PAL itself, so they link here.
if(WIN32)
  target_link_libraries(q35 PUBLIC ws2_32 psapi)
endif()

enable_testing()

add_executable(test_tok tests/test_tok.c)
target_link_libraries(test_tok PRIVATE q35)
target_include_directories(test_tok PRIVATE tests)
add_test(NAME tok COMMAND test_tok ${CMAKE_SOURCE_DIR}/tokenizer/tokenizer.bin)
add_executable(test_bpe tests/test_bpe.c)
target_link_libraries(test_bpe PRIVATE q35)
add_test(NAME bpe COMMAND test_bpe ${CMAKE_SOURCE_DIR}/tokenizer/tokenizer.bin ${CMAKE_SOURCE_DIR}/tests/fixtures/encode_fixtures.txt)
set_tests_properties(bpe PROPERTIES SKIP_RETURN_CODE 77)

add_executable(q35cli src/q35_cli.c src/cli/run_cmd.c src/cli/serve_cmd.c src/cli/cli_common.c)
set_target_properties(q35cli PROPERTIES OUTPUT_NAME qwen35)
target_link_libraries(q35cli PRIVATE q35)
target_include_directories(q35cli PRIVATE src)

add_executable(test_st tests/test_st.c)
target_link_libraries(test_st PRIVATE q35)
add_test(NAME st COMMAND test_st ${CMAKE_SOURCE_DIR}/tests/fixtures/st ${CMAKE_SOURCE_DIR}/../Qwen3.8-27B-FP8)
set_tests_properties(st PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_mm tests/test_mm.c)
target_link_libraries(test_mm PRIVATE q35)
add_test(NAME mm COMMAND test_mm ${CMAKE_SOURCE_DIR}/tests/fixtures/mm)

add_executable(test_kern tests/test_kern.c)
target_link_libraries(test_kern PRIVATE q35)
target_include_directories(test_kern PRIVATE src)
add_test(NAME kern COMMAND test_kern ${CMAKE_SOURCE_DIR}/tests/fixtures/kern)
set_tests_properties(kern PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_deltanet tests/test_deltanet.c)
target_link_libraries(test_deltanet PRIVATE q35)
target_include_directories(test_deltanet PRIVATE src)
add_test(NAME deltanet COMMAND test_deltanet ${CMAKE_SOURCE_DIR}/tests/fixtures/deltanet)
set_tests_properties(deltanet PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_attn tests/test_attn.c)
target_link_libraries(test_attn PRIVATE q35)
add_test(NAME attn COMMAND test_attn ${CMAKE_SOURCE_DIR}/tests/fixtures/attn)
set_tests_properties(attn PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_model tests/test_model.c)
target_link_libraries(test_model PRIVATE q35)
target_include_directories(test_model PRIVATE src)
add_test(NAME model COMMAND test_model ${CMAKE_SOURCE_DIR}/tests/fixtures/model)
set_tests_properties(model PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_oracle tests/test_oracle.c)
target_link_libraries(test_oracle PRIVATE q35)
add_test(NAME oracle COMMAND test_oracle ${CMAKE_SOURCE_DIR}/tests/fixtures/oracle)
set_tests_properties(oracle PROPERTIES SKIP_RETURN_CODE 77)

add_executable(dbg_reset tools/dbg_reset.c)
target_link_libraries(dbg_reset PRIVATE q35)

add_executable(dump_real tools/dump_real.c)
target_link_libraries(dump_real PRIVATE q35)

add_executable(run_regression tools/run_regression.c)
target_link_libraries(run_regression PRIVATE q35)
target_include_directories(run_regression PRIVATE src)

add_executable(test_batch_mm tests/test_batch_mm.c)
target_link_libraries(test_batch_mm PRIVATE q35)
add_test(NAME batch_mm COMMAND test_batch_mm)

add_executable(test_prefill tests/test_prefill.c)
target_link_libraries(test_prefill PRIVATE q35)
add_test(NAME prefill COMMAND test_prefill ${CMAKE_SOURCE_DIR}/tests/fixtures/model)

add_executable(test_state tests/test_state.c)
target_link_libraries(test_state PRIVATE q35)
add_test(NAME state COMMAND test_state ${CMAKE_SOURCE_DIR}/tests/fixtures/model)

