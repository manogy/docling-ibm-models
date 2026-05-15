# ZDLC ConfigMap Configuration Guide

This guide explains the environment variables used to control ZDLC (IBM Z Deep Learning Compiler) backend behavior in Docling IBM Models.

## Environment Variables

### DOCLING_DISABLE_ZDLC

**Master disable switch for ZDLC backend**

- **Type**: Boolean string
- **Valid values**: `"true"` or `"false"` (case-insensitive)
- **Default**: `"false"`

**Purpose**: Controls whether ZDLC backend can be used at all, regardless of architecture.

**Behavior**:
- `"false"` (default): Allows automatic ZDLC usage on s390x systems when available
- `"true"`: Forces PyTorch backend on all architectures, including s390x

**When to set to "true"**:
- Troubleshooting ZDLC-related issues
- Performance comparison between ZDLC and PyTorch
- Testing PyTorch compatibility on s390x
- Ensuring consistent behavior across different architectures

---

### DOCLING_ZDLC_LAYOUT_PREDICTOR

**Fine-grained control for ZDLC in LayoutPredictor model**

- **Type**: Boolean string
- **Valid values**: `"true"` or `"false"` (case-insensitive)
- **Default**: `"false"`

**Purpose**: Enables ZDLC specifically for the layout detection model, which identifies document elements like paragraphs, tables, figures, etc.

**Prerequisites**:
- Must be running on s390x architecture
- `DOCLING_DISABLE_ZDLC` must NOT be `"true"`
- `zdlc_pyrt` package must be installed

**Behavior**:
- `"false"` (default): Uses PyTorch for LayoutPredictor
- `"true"`: Uses ZDLC for LayoutPredictor on s390x (if available)

**Performance note**: ZDLC typically provides better performance on IBM Z hardware.

---

### DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER

**Fine-grained control for ZDLC in DocumentFigureClassifier model**

- **Type**: Boolean string
- **Valid values**: `"true"` or `"false"` (case-insensitive)
- **Default**: `"false"`

**Purpose**: Enables ZDLC specifically for the figure classification model, which categorizes document figures (charts, diagrams, maps, etc.).

**Prerequisites**:
- Must be running on s390x architecture
- `DOCLING_DISABLE_ZDLC` must NOT be `"true"`
- `zdlc_pyrt` package must be installed

**Behavior**:
- `"false"` (default): Uses PyTorch for DocumentFigureClassifier
- `"true"`: Uses ZDLC for DocumentFigureClassifier on s390x (if available)

**Performance note**: ZDLC typically provides better performance on IBM Z hardware.

---

## ConfigMap Example

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: docling-zdlc-config
  namespace: your-namespace
data:
  # Master disable switch - set to "true" to force PyTorch on all architectures
  # Default: "false" - allows ZDLC on s390x when available
  DOCLING_DISABLE_ZDLC: "false"

  # Enable ZDLC for LayoutPredictor on s390x
  # Default: "false" - uses PyTorch
  # Set to "true" to enable ZDLC for layout detection
  DOCLING_ZDLC_LAYOUT_PREDICTOR: "false"

  # Enable ZDLC for DocumentFigureClassifier on s390x
  # Default: "false" - uses PyTorch
  # Set to "true" to enable ZDLC for figure classification
  DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER: "false"
```

## Usage in Kubernetes Deployment

### Method 1: Using envFrom (Recommended)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docling-service
spec:
  template:
    spec:
      containers:
      - name: docling
        image: your-docling-image:latest
        envFrom:
        - configMapRef:
            name: docling-zdlc-config
```

### Method 2: Using individual env entries

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: docling-service
spec:
  template:
    spec:
      containers:
      - name: docling
        image: your-docling-image:latest
        env:
        - name: DOCLING_DISABLE_ZDLC
          valueFrom:
            configMapKeyRef:
              name: docling-zdlc-config
              key: DOCLING_DISABLE_ZDLC
        - name: DOCLING_ZDLC_LAYOUT_PREDICTOR
          valueFrom:
            configMapKeyRef:
              name: docling-zdlc-config
              key: DOCLING_ZDLC_LAYOUT_PREDICTOR
        - name: DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER
          valueFrom:
            configMapKeyRef:
              name: docling-zdlc-config
              key: DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER
```

## Configuration Scenarios

### Scenario 1: Default (PyTorch on all architectures)

```yaml
DOCLING_DISABLE_ZDLC: "false"
DOCLING_ZDLC_LAYOUT_PREDICTOR: "false"
DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER: "false"
```

**Result**: Uses PyTorch backend on all architectures

---

### Scenario 2: Enable ZDLC for all models on s390x

```yaml
DOCLING_DISABLE_ZDLC: "false"
DOCLING_ZDLC_LAYOUT_PREDICTOR: "true"
DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER: "true"
```

**Result**:
- On s390x: Uses ZDLC for both models (if available)
- On other architectures: Uses PyTorch

---

### Scenario 3: Force PyTorch even on s390x

```yaml
DOCLING_DISABLE_ZDLC: "true"
DOCLING_ZDLC_LAYOUT_PREDICTOR: "false"  # Ignored when master switch is true
DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER: "false"  # Ignored when master switch is true
```

**Result**: Uses PyTorch on all architectures, including s390x

---

### Scenario 4: Selective ZDLC usage (only LayoutPredictor)

```yaml
DOCLING_DISABLE_ZDLC: "false"
DOCLING_ZDLC_LAYOUT_PREDICTOR: "true"
DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER: "false"
```

**Result**:
- LayoutPredictor uses ZDLC on s390x
- DocumentFigureClassifier uses PyTorch on all architectures

---

## Decision Flow

```
Is DOCLING_DISABLE_ZDLC = "true"?
├─ YES → Use PyTorch (all models, all architectures)
└─ NO → Continue...
    │
    Is architecture s390x?
    ├─ NO → Use PyTorch (all models)
    └─ YES → Continue...
        │
        Is zdlc_pyrt installed?
        ├─ NO → Use PyTorch (fallback with warning)
        └─ YES → Check per-model settings:
            │
            ├─ LayoutPredictor:
            │   DOCLING_ZDLC_LAYOUT_PREDICTOR = "true" → ZDLC
            │   DOCLING_ZDLC_LAYOUT_PREDICTOR = "false" → PyTorch
            │
            └─ DocumentFigureClassifier:
                DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER = "true" → ZDLC
                DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER = "false" → PyTorch
```

## Applying the ConfigMap

```bash
# Create the ConfigMap
kubectl apply -f docling-zdlc-configmap.yaml

# Verify the ConfigMap
kubectl get configmap docling-zdlc-config -o yaml

# Update existing deployment to use the ConfigMap
kubectl set env deployment/your-deployment --from=configmap/docling-zdlc-config

# Restart pods to pick up new configuration
kubectl rollout restart deployment/your-deployment
```

## Troubleshooting

### ZDLC not being used on s390x

1. Check if `DOCLING_DISABLE_ZDLC` is set to `"true"`
2. Verify the per-model environment variable is set to `"true"`
3. Ensure `zdlc_pyrt` package is installed
4. Check application logs for ZDLC availability messages

### Verifying backend in use

Check the application logs at startup. You should see messages like:

```
INFO - Running on s390x architecture - ZDLC available
```

or

```
INFO - Running on x86_64 architecture - PyTorch backend will be used
```

### Performance issues

If experiencing performance issues:
1. Try enabling ZDLC on s390x by setting model-specific variables to `"true"`
2. Compare performance with PyTorch by setting `DOCLING_DISABLE_ZDLC: "true"`
3. Monitor resource usage (CPU, memory) with both backends

## Best Practices

1. **Start with defaults**: Use PyTorch initially (`"false"` for all variables)
2. **Test incrementally**: Enable ZDLC for one model at a time
3. **Monitor performance**: Compare metrics before and after enabling ZDLC
4. **Document your choice**: Add comments in your ConfigMap explaining why specific values were chosen
5. **Version control**: Keep ConfigMaps in version control alongside your deployment manifests
