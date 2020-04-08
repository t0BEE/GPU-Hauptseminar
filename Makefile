
LATEXMK ?= latexmk

SRC = ausarbeitung.tex

all:
	@[ -x "`which $(LATEXMK)`" ] || { printf "We suggest using latexmk for building.\\nLatexmk not found.\\n" >&2; exit 1; }
	$(LATEXMK) -pdf $(SRC)

clean:
	$(LATEXMK) -c

.PHONY: all clean

