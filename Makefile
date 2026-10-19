.PHONY: test test-python test-rust test-consistency site report

test: test-python test-rust test-consistency

test-python:
	@set -e; for t in python/*/test_*.py; do echo "== $$t"; python3 "$$t"; done

test-rust:
	cd rust && cargo test --quiet

test-consistency:
	python3 scripts/check_consistency.py

site:
	python3 scripts/build_site.py

report:
	python3 scripts/report.py
