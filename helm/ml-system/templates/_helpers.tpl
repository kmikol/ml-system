{{- define "ml-system.labels" -}}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
