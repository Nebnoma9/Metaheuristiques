#!/bin/sh
pandoc --defaults docs/config_pdf.yaml --webtex --pdf-engine=xelatex -o LESSON.pdf LESSON.md
