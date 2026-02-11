# Minimal makefile for Sphinx documentation, compatible with RTD
#

# Directories
SOURCEDIR     = source
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build

# Detect ReadTheDocs environment
ifdef READTHEDOCS
BUILDDIR = $(READTHEDOCS_OUTPUT)
else
BUILDDIR = _build
endif

# Default help target
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile
