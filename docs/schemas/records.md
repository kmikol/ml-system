# Event and Record Schemas

These models represent internal data records that flow through storage, monitoring, and retraining loops.

Used primarily by:

- prediction/event persistence
- annotation and drift workflows
- model monitoring and retraining pipeline stages

--8<-- "docs/schemas/generated/records-fields.md"

## Module Reference

### PredictRecord

::: shared.schemas.predict_record
		options:
			show_source: false
			show_root_full_path: false
			show_root_toc_entry: false

### InferenceEvent

::: shared.schemas.inference_event
		options:
			show_source: false
			show_root_full_path: false
			show_root_toc_entry: false

::: shared.schemas.feature_schema
		options:
			show_source: false
			show_root_full_path: false
			show_root_toc_entry: false
