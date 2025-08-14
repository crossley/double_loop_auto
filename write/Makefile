# Makefile to build the LaTeX document

# Document name without extension
DOC = main

# Compiler
LATEX = xelatex

# Compilation flags (nonstopmode ensures compilation continues after errors)
FLAGS = -interaction=nonstopmode -halt-on-error

# List of intermediate file extensions to clean
INTERMEDIATE_EXTENSIONS = aux log out toc lof lot

# Default target: build PDF
all: $(DOC).pdf

# Rule to build PDF
$(DOC).pdf: $(DOC).tex
	$(LATEX) $(FLAGS) $(DOC).tex
	$(LATEX) $(FLAGS) $(DOC).tex  # Run twice for cross-references

# Clean intermediate files
clean:
	rm -f $(foreach ext,$(INTERMEDIATE_EXTENSIONS),$(DOC).$(ext))

# Clean everything including PDF
distclean: clean
	rm -f $(DOC).pdf

