# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0

target_sources(onnx_core PRIVATE
    attr_proto_util.cc
    attr_proto_util.h
    data_propagators.h
    data_type_utils.cc
    data_type_utils.h
    doc_strings.cc
    doc_strings.h
    function.cc
    function.h
    operator_sets.h
    operator_sets_ml.h
    operator_sets_preview.h
    operator_sets_training.h
    parser.cc
    parser.h
    printer.cc
    printer.h
    schema.cc
    schema.h
    shape_inference.cc
    shape_inference.h
    tensor_proto_util.cc
    tensor_proto_util.h
    tensor_util.cc
    tensor_util.h
)

set(OP_SPECS_GENERATED_DIR "${CMAKE_BINARY_DIR}/onnx/defs/generated")
set(OP_SPECS_GENERATED_H "${OP_SPECS_GENERATED_DIR}/op_specs_generated.h")
set(OP_SPECS_GENERATED_CC "${OP_SPECS_GENERATED_DIR}/op_specs_generated.cc")
add_custom_command(
    OUTPUT ${OP_SPECS_GENERATED_H} ${OP_SPECS_GENERATED_CC}
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/specs/tools/gen_cpp.py"
            --output-dir "${OP_SPECS_GENERATED_DIR}"
    DEPENDS "${CMAKE_SOURCE_DIR}/specs/tools/gen_cpp.py"
            "${CMAKE_SOURCE_DIR}/specs/tools/models.py"
            # "${CMAKE_SOURCE_DIR}/specs/tools/build_docs.py"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/specs/tools"
    COMMENT "Generating operator schema C++ from YAML specs"
    VERBATIM
)
add_custom_target(gen_op_specs DEPENDS ${OP_SPECS_GENERATED_H} ${OP_SPECS_GENERATED_CC})
add_dependencies(onnx_core gen_op_specs)
target_sources(onnx_core PRIVATE ${OP_SPECS_GENERATED_CC})
target_include_directories(onnx_core PRIVATE "${CMAKE_BINARY_DIR}")

add_subdirectory(controlflow)
add_subdirectory(generator)
add_subdirectory(image)
add_subdirectory(logical)
add_subdirectory(math)
add_subdirectory(nn)
add_subdirectory(object_detection)
add_subdirectory(optional)
add_subdirectory(preview)
add_subdirectory(quantization)
add_subdirectory(reduction)
add_subdirectory(rnn)
add_subdirectory(sequence)
add_subdirectory(tensor)
add_subdirectory(text)
add_subdirectory(traditionalml)
add_subdirectory(training)

# These files exhibit issues in unity builds, so we disable them.
set_source_files_properties(schema.cc tensor_proto_util.cc
    TARGET_DIRECTORY onnx_core
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION TRUE
)
