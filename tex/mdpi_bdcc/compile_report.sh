#!/bin/bash
set -e

# Script is running inside tex/mdpi_bdcc/ folder
export PATH="/Library/TeX/texbin:$PATH"
export TEXINPUTS="./Definitions//:${TEXINPUTS}"

echo "--- Building Report: Pass 1 (pdflatex) ---"
pdflatex -interaction=nonstopmode report.tex

echo "--- Building Bibliography (bibtex) ---"
bibtex report

echo "--- Building Report: Pass 2 (pdflatex) ---"
pdflatex -interaction=nonstopmode report.tex

echo "--- Building Report: Final Pass (pdflatex) ---"
pdflatex -interaction=nonstopmode report.tex

if [ -f "report.pdf" ]; then
    echo "--- Build Successful!"
else
    echo "Error: report.pdf was not generated."
    exit 1
fi
