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

## Validation Steps

### Step 1: Verify Environment Variables

Check that your environment variables are correctly set in the pod:

```bash
# Get pod name
POD_NAME=$(kubectl get pods -l app=docling-service -o jsonpath='{.items[0].metadata.name}')

# Check environment variables
kubectl exec $POD_NAME -- env | grep DOCLING
```

Expected output:
```
DOCLING_DISABLE_ZDLC=false
DOCLING_ZDLC_LAYOUT_PREDICTOR=true
DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER=true
```

---

### Step 2: Check Application Logs for Backend Detection

View the application logs to verify which backend is being used:

```bash
# View logs
kubectl logs $POD_NAME | grep -E "(s390x|ZDLC|PyTorch|architecture)"
```

**Expected log messages on s390x with ZDLC enabled:**

```
INFO - Running on s390x architecture - ZDLC available
INFO - LayoutPredictor: Using ZDLC backend
INFO - DocumentFigureClassifier: Using ZDLC backend
```

**Expected log messages on s390x with ZDLC disabled:**

```
INFO - Running on s390x architecture - PyTorch backend will be used
INFO - LayoutPredictor: Using PyTorch backend
INFO - DocumentFigureClassifier: Using PyTorch backend
```

**Expected log messages on non-s390x architectures:**

```
INFO - Running on x86_64 architecture - PyTorch backend will be used
INFO - LayoutPredictor: Using PyTorch backend
INFO - DocumentFigureClassifier: Using PyTorch backend
```

---

### Step 3: Verify ZDLC Package Installation (s390x only)

If running on s390x, verify that the ZDLC package is installed:

```bash
# Check if zdlc_pyrt is installed
kubectl exec $POD_NAME -- python -c "import zdlc_pyrt; print('ZDLC version:', zdlc_pyrt.__version__)"
```

**Expected output if installed:**
```
ZDLC version: <version-number>
```

**Expected output if not installed:**
```
ModuleNotFoundError: No module named 'zdlc_pyrt'
```

If not installed, you'll see a warning in logs:
```
WARNING - Running on s390x but zdlc_pyrt not available, falling back to PyTorch
```

---

### Step 4: Test Model Inference

Run a simple inference test to verify the models are working:

```bash
# Create a test script
cat <<'EOF' > test_zdlc.py
import platform
import os
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import DocumentFigureClassifierPredictor

print(f"Architecture: {platform.machine()}")
print(f"DOCLING_DISABLE_ZDLC: {os.environ.get('DOCLING_DISABLE_ZDLC', 'not set')}")
print(f"DOCLING_ZDLC_LAYOUT_PREDICTOR: {os.environ.get('DOCLING_ZDLC_LAYOUT_PREDICTOR', 'not set')}")
print(f"DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER: {os.environ.get('DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER', 'not set')}")

# Initialize models
print("\nInitializing LayoutPredictor...")
layout_predictor = LayoutPredictor()
print("✓ LayoutPredictor initialized successfully")

print("\nInitializing DocumentFigureClassifier...")
classifier = DocumentFigureClassifierPredictor()
print("✓ DocumentFigureClassifier initialized successfully")

print("\n✓ All models initialized successfully with configured backend")
EOF

# Copy and run the test
kubectl cp test_zdlc.py $POD_NAME:/tmp/test_zdlc.py
kubectl exec $POD_NAME -- python /tmp/test_zdlc.py
```

**Expected output:**
```
Architecture: s390x
DOCLING_DISABLE_ZDLC: false
DOCLING_ZDLC_LAYOUT_PREDICTOR: true
DOCLING_ZDLC_DOCUMENT_FIGURE_CLASSIFIER: true

Initializing LayoutPredictor...
INFO - Running on s390x architecture - ZDLC available
✓ LayoutPredictor initialized successfully

Initializing DocumentFigureClassifier...
INFO - Running on s390x architecture - ZDLC available
✓ DocumentFigureClassifier initialized successfully

✓ All models initialized successfully with configured backend
```

---

### Step 5: Performance Validation

Compare inference times between ZDLC and PyTorch backends:

```bash
# Test with ZDLC enabled
kubectl set env deployment/docling-service DOCLING_ZDLC_LAYOUT_PREDICTOR=true
kubectl rollout status deployment/docling-service
# Run your performance tests and record metrics

# Test with PyTorch (disable ZDLC)
kubectl set env deployment/docling-service DOCLING_ZDLC_LAYOUT_PREDICTOR=false
kubectl rollout status deployment/docling-service
# Run the same performance tests and compare
```

**Metrics to monitor:**
- Inference latency (ms per document)
- Throughput (documents per second)
- CPU utilization
- Memory usage

---

### Step 6: Validate Configuration Changes

After changing ConfigMap values, verify the changes are applied:

```bash
# Update ConfigMap
kubectl edit configmap docling-zdlc-config

# Restart deployment to pick up changes
kubectl rollout restart deployment/docling-service

# Wait for rollout to complete
kubectl rollout status deployment/docling-service

# Verify new environment variables
POD_NAME=$(kubectl get pods -l app=docling-service -o jsonpath='{.items[0].metadata.name}')
kubectl exec $POD_NAME -- env | grep DOCLING

# Check logs for new backend selection
kubectl logs $POD_NAME | grep -E "(ZDLC|PyTorch|backend)"
```

---

## Troubleshooting

### ZDLC not being used on s390x

**Symptoms:**
- Logs show "PyTorch backend will be used" on s390x
- Expected ZDLC performance improvements not observed

**Diagnostic steps:**

1. **Check master disable switch:**
   ```bash
   kubectl exec $POD_NAME -- env | grep DOCLING_DISABLE_ZDLC
   ```
   Should be `"false"` or not set

2. **Verify per-model environment variable:**
   ```bash
   kubectl exec $POD_NAME -- env | grep DOCLING_ZDLC_LAYOUT_PREDICTOR
   ```
   Should be `"true"` to enable ZDLC

3. **Ensure zdlc_pyrt package is installed:**
   ```bash
   kubectl exec $POD_NAME -- python -c "import zdlc_pyrt; print('OK')"
   ```
   Should print "OK" without errors

4. **Check application logs for warnings:**
   ```bash
   kubectl logs $POD_NAME | grep -i "zdlc\|warning\|error"
   ```

**Common issues:**
- `DOCLING_DISABLE_ZDLC` is set to `"true"` → Set to `"false"`
- Per-model variable not set to `"true"` → Update ConfigMap
- `zdlc_pyrt` not installed → Install package in container image
- Wrong architecture detected → Verify with `platform.machine()`

---

### Verifying backend in use

**Method 1: Check startup logs**

```bash
kubectl logs $POD_NAME | head -50 | grep -E "(architecture|ZDLC|PyTorch)"
```

Expected messages:
- **ZDLC enabled:** `INFO - Running on s390x architecture - ZDLC available`
- **ZDLC disabled:** `INFO - Running on s390x architecture - PyTorch backend will be used`
- **Non-s390x:** `INFO - Running on x86_64 architecture - PyTorch backend will be used`

**Method 2: Runtime verification**

```bash
kubectl exec $POD_NAME -- python -c "
from docling_ibm_models.layoutmodel import layout_predictor
print('ZDLC Available:', layout_predictor._ZDLC_AVAILABLE)
print('Architecture:', layout_predictor._IS_S390X)
print('Disable ZDLC:', layout_predictor._DOCLING_DISABLE_ZDLC)
print('Use ZDLC Layout:', layout_predictor._USE_ZDLC_LAYOUT_PREDICTOR)
"
```

---

### Performance issues

**Symptoms:**
- Slower than expected inference times
- High CPU or memory usage
- Inconsistent performance

**Diagnostic steps:**

1. **Compare ZDLC vs PyTorch performance:**
   ```bash
   # Enable ZDLC
   kubectl set env deployment/docling-service DOCLING_ZDLC_LAYOUT_PREDICTOR=true
   # Run benchmarks and record metrics

   # Disable ZDLC
   kubectl set env deployment/docling-service DOCLING_DISABLE_ZDLC=true
   # Run same benchmarks and compare
   ```

2. **Monitor resource usage:**
   ```bash
   kubectl top pod $POD_NAME
   ```

3. **Check for errors in logs:**
   ```bash
   kubectl logs $POD_NAME | grep -i "error\|exception\|failed"
   ```

**Solutions:**
- If ZDLC is slower: Verify `zdlc_pyrt` version compatibility
- If PyTorch is slower on s390x: Enable ZDLC with `DOCLING_ZDLC_LAYOUT_PREDICTOR=true`
- If memory issues: Adjust pod resource limits
- If CPU issues: Check for proper CPU affinity on s390x

---

### ConfigMap changes not taking effect

**Symptoms:**
- Updated ConfigMap but environment variables unchanged
- Old backend still being used after update

**Solution:**

```bash
# Verify ConfigMap was updated
kubectl get configmap docling-zdlc-config -o yaml

# Restart deployment to pick up changes
kubectl rollout restart deployment/docling-service

# Wait for new pods
kubectl rollout status deployment/docling-service

# Verify in new pod
POD_NAME=$(kubectl get pods -l app=docling-service -o jsonpath='{.items[0].metadata.name}')
kubectl exec $POD_NAME -- env | grep DOCLING
```

**Note:** Pods must be restarted for ConfigMap changes to take effect.

## Best Practices

1. **Start with defaults**: Use PyTorch initially (`"false"` for all variables)
2. **Test incrementally**: Enable ZDLC for one model at a time
3. **Monitor performance**: Compare metrics before and after enabling ZDLC
4. **Document your choice**: Add comments in your ConfigMap explaining why specific values were chosen
5. **Version control**: Keep ConfigMaps in version control alongside your deployment manifests
