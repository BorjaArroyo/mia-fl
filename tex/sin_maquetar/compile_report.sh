#!/bin/bash
set -e

# Script is running inside tex/ folder

echo "--- Building Report: Pass 1 (pdflatex) ---"
pdflatex report.tex

echo "--- Building Bibliography (bibtex) ---"
bibtex report

echo "--- Building Report: Pass 2 (pdflatex) ---"
pdflatex report.tex

echo "--- Building Report: Final Pass (pdflatex) ---"
pdflatex report.tex

if [ -f "report.pdf" ]; then
    echo "--- Build Successful!"
else
    echo "Error: report.pdf was not generated."
    exit 1
fi
