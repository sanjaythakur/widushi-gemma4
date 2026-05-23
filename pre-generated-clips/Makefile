PIPER_IMAGE := piper-clips
INDIC_IMAGE := indic-clips

# Where the api-preset wavs land. Default points at the gemma-llama
# api-presets bind mount, which the api container reads read-only at
# /api/presets (see ../gemma-llama/docker-compose.yml). Override on the
# command line if you want the wavs to land elsewhere -- e.g.
#   make presets PRESETS_OUT_DIR=$$PWD/_scratch
PRESETS_OUT_DIR ?= $(CURDIR)/../gemma-llama/api-presets

.PHONY: build build-piper build-indic \
        clips clips-en clips-hi \
        presets presets-en presets-hi \
        clean-clips clean-clips-en clean-clips-hi \
        clean-presets

build: build-piper build-indic

build-piper:
	docker build -t $(PIPER_IMAGE) -f docker/Dockerfile.piper .

build-indic:
	docker build -t $(INDIC_IMAGE) -f docker/Dockerfile.indic .

# ---------------------------------------------------------------------------
# Phase-1 acknowledgement / status clips (originals)
# ---------------------------------------------------------------------------

clips: clips-en clips-hi

clips-en: build-piper
	docker run --rm -v "$$PWD:/work" $(PIPER_IMAGE)

clips-hi: build-indic
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: HF_TOKEN is not set. The Indic-Parler-TTS model is gated."; \
		echo "  1. Accept agreement: https://huggingface.co/ai4bharat/indic-parler-tts"; \
		echo "  2. Create a read token: https://huggingface.co/settings/tokens"; \
		echo "  3. Re-run as: HF_TOKEN=hf_xxx make clips-hi"; \
		exit 1; \
	fi
	docker run --rm -v "$$PWD:/work" -e HF_TOKEN="$$HF_TOKEN" $(INDIC_IMAGE)

clean-clips: clean-clips-en clean-clips-hi

clean-clips-en:
	rm -rf clips/en
	mkdir -p clips/en

clean-clips-hi:
	rm -rf clips/hi
	mkdir -p clips/hi

# ---------------------------------------------------------------------------
# gemma-llama api-presets playground audio
#
# Generates the wavs referenced by ../gemma-llama/api-presets/manifest.json.
# Reuses the existing piper-clips / indic-clips images and overrides the
# entrypoint to run scripts/generate_presets_{en,hi}.py from the bind
# mount. PRESETS_OUT_DIR defaults to the gemma-llama api-presets folder
# so the wavs land where the api container expects them.
# ---------------------------------------------------------------------------

presets: presets-en presets-hi

presets-en: build-piper
	@mkdir -p "$(PRESETS_OUT_DIR)"
	docker run --rm \
		-v "$$PWD:/work" \
		-v "$(PRESETS_OUT_DIR):/out" \
		-e OUT_DIR=/out \
		--entrypoint python \
		$(PIPER_IMAGE) /work/scripts/generate_presets_en.py

presets-hi: build-indic
	@if [ -z "$$HF_TOKEN" ]; then \
		echo "ERROR: HF_TOKEN is not set. The Indic-Parler-TTS model is gated."; \
		echo "  1. Accept agreement: https://huggingface.co/ai4bharat/indic-parler-tts"; \
		echo "  2. Create a read token: https://huggingface.co/settings/tokens"; \
		echo "  3. Re-run as: HF_TOKEN=hf_xxx make presets-hi"; \
		exit 1; \
	fi
	@mkdir -p "$(PRESETS_OUT_DIR)"
	docker run --rm \
		-v "$$PWD:/work" \
		-v "$(PRESETS_OUT_DIR):/out" \
		-e OUT_DIR=/out \
		-e HF_TOKEN="$$HF_TOKEN" \
		--entrypoint python \
		$(INDIC_IMAGE) /work/scripts/generate_presets_hi.py

# Wipes only the wavs this Makefile generates, not the gemma-llama
# manifest.json or any image files dropped in by hand.
clean-presets:
	rm -f "$(PRESETS_OUT_DIR)"/free-convo/*.wav
	rm -f "$(PRESETS_OUT_DIR)"/voice-mirror-score/*.wav
	rm -f "$(PRESETS_OUT_DIR)"/vision-teach/*.wav
	rm -f "$(PRESETS_OUT_DIR)"/audio-listen/*.wav
