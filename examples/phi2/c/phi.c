#include <ort_genai_c.h>

OgaModel* model;
CheckResult(OgaCreateModel(MODEL_PATH "example-models/phi2-int4-cpu", OgaDeviceTypeCPU, &model));
OgaModelPtr model_ptr{model};

OgaTokenizer* tokenizer;
CheckResult(OgaCreateTokenizer(model, &tokenizer));
OgaTokenizerPtr tokenizer_ptr{tokenizer};

const char* input_strings[] = {
    "This is a test.",
    "Rats are awesome pets!",
    "The quick brown fox jumps over the lazy dog.",
};

OgaSequences* input_sequences;
CheckResult(OgaTokenizerEncodeBatch(tokenizer, input_strings, std::size(input_strings), &input_sequences));
OgaSequencesPtr sequences_ptr{input_sequences};

OgaGeneratorParams* params;
CheckResult(OgaCreateGeneratorParams(model, &params));
OgaGeneratorParamsPtr params_ptr{params};
CheckResult(OgaGeneratorParamsSetMaxLength(params, 20));
CheckResult(OgaGeneratorParamsSetInputSequences(params, input_sequences));

OgaSequences* output_sequences;
CheckResult(OgaGenerate(model, params, &output_sequences));
OgaSequencesPtr output_sequences_ptr{output_sequences};

// Decode The Batch

const char** out_strings;
CheckResult(OgaTokenizerDecodeBatch(tokenizer, output_sequences, &out_strings));
for (size_t i = 0; i < OgaSequencesCount(output_sequences); i++) {
    std::cout << "Decoded string:" << out_strings[i] << std::endl;
}
OgaTokenizerDestroyStrings(out_strings, OgaSequencesCount(output_sequences));
