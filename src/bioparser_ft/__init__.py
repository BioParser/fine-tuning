"""BioParser fine-tuning toolkit.

Pipeline::

    client PDF --MinerU--> markdown --clean_markdown--> prompt --Qwen3.5+LoRA--> model TSV
        --postprocess.to_ets_records--> list[dict] (JSON) --> FastAPI response

Training pairs are (parsed PDF markdown, client ETS TSV).

Module overview (stdlib-only modules can be vendored into the FastAPI service):

* ``schema``      - column contract, client field definitions, prompt   (stdlib only)
* ``markdown``    - deterministic cleaning of parser output             (stdlib only)
* ``tsv``         - client ETS TSV reader, model TSV writer/parser       (stdlib only)
* ``postprocess`` - model TSV -> ETS JSON records                       (stdlib only)
* ``coverage``    - checks that a parsed PDF really contains the labels  (stdlib only)
* ``linking``     - pairs ETS TSV files with parsed PDFs                 (stdlib only)
* ``evaluate``    - field-level metrics                                  (stdlib only)
* ``build_dataset`` - CLI: builds train/val/test JSONL                   (stdlib only)
* ``chat``        - chat-template rendering shared by train + inference (needs a tokenizer)
* ``train`` / ``infer`` - Unsloth training and generation               (GPU stack)
"""

__version__ = "0.2.0"
