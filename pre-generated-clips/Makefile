PIPER_IMAGE := piper-clips
INDIC_IMAGE := indic-clips

.PHONY: build build-piper build-indic clips clips-en clips-hi clean-clips clean-clips-en clean-clips-hi

build: build-piper build-indic

build-piper:
	docker build -t $(PIPER_IMAGE) -f docker/Dockerfile.piper .

build-indic:
	docker build -t $(INDIC_IMAGE) -f docker/Dockerfile.indic .

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
