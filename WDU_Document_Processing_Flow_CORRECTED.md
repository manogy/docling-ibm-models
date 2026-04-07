# Watson Document Understanding - End-to-End Document Processing Flow (CORRECTED)

## Mermaid Diagram

```mermaid
graph TB
    Start([Document Input<br/>PDF/Image/DOCX]) --> WDU[Watson Document Understanding Service]
    
    WDU --> APIGateway[API Gateway<br/>REST/gRPC]
    APIGateway --> InitialProcessor[Initial Processor Worker<br/>Document Validation]
    
    InitialProcessor --> JobQueue[Job Queue]
    
    JobQueue --> PageServer[Page Server Worker<br/>Monitors Page Queue]
    
    PageServer --> PageWorker[Page Worker<br/>Core Document Processing<br/>Loads ALL Models]
    
    PageWorker --> ModelsProvider[Models Provider<br/>ModelsProvider.load_all]
    
    ModelsProvider --> OCRPath{OCR Models}
    ModelsProvider --> LayoutPath{Layout Models}
    ModelsProvider --> TablePath{Table Models}
    ModelsProvider --> FigurePath{Figure Models}
    ModelsProvider --> CodePath{Code/Formula Models}
    
    OCRPath --> IOCR[IOCR Models<br/>hrl_ocr package]
    IOCR --> TextDetection[Text Detection Model]
    IOCR --> OCRRecognition[OCR Recognition Model]
    
    TextDetection --> IOCROutput[IOCR Output<br/>- Text Lines<br/>- Bounding Boxes<br/>- Blocks]
    OCRRecognition --> IOCROutput
    
    LayoutPath --> DoclingLayout[Docling Layout Model<br/>docling-ibm-models]
    DoclingLayout --> LayoutPredictor[Layout Predictor]
    LayoutPredictor --> LayoutZDLC{Use ZDLC?}
    
    LayoutZDLC -->|Yes| LayoutZDLCPred[Layout Predictor ZDLC<br/>layout_predictor_zdlc.py<br/>.so model]
    LayoutZDLC -->|No| LayoutPyTorch[Layout Predictor PyTorch<br/>layout_predictor.py<br/>.pt model]
    
    LayoutZDLCPred --> LayoutOutput[Layout Output<br/>- Page Elements<br/>- Sections<br/>- Headers/Footers]
    LayoutPyTorch --> LayoutOutput
    
    TablePath --> DoclingTable[Docling Table Models<br/>docling-ibm-models]
    DoclingTable --> TableFormer[TableFormer]
    TableFormer --> TableZDLC{Use ZDLC?}
    
    TableZDLC -->|Yes| TableZDLCPred[TableFormer ZDLC<br/>tf_predictor_zdlc.py<br/>.so model]
    TableZDLC -->|No| TablePyTorch[TableFormer PyTorch<br/>tf_predictor.py<br/>.pt model]
    
    TableZDLCPred --> TableOutput[Table Output<br/>- Table Structure<br/>- Cells<br/>- Rows/Columns]
    TablePyTorch --> TableOutput
    
    FigurePath --> FigureClassifier[Figure Classifier<br/>document_figure_classifier]
    FigureClassifier --> FigureZDLC{Use ZDLC?}
    
    FigureZDLC -->|Yes| FigureZDLCPred[Figure Classifier ZDLC<br/>.so model]
    FigureZDLC -->|No| FigurePyTorch[Figure Classifier PyTorch<br/>.pt model]
    
    FigureZDLCPred --> FigureOutput[Figure Classification<br/>- Chart Types<br/>- Diagrams]
    FigurePyTorch --> FigureOutput
    
    CodePath --> CodeFormula[Code/Formula Detector<br/>code_formula_predictor]
    CodeFormula --> CodeFormulaZDLC{Use ZDLC?}
    
    CodeFormulaZDLC -->|Yes| CodeFormulaZDLCPred[Code/Formula ZDLC<br/>.so model]
    CodeFormulaZDLC -->|No| CodeFormulaPyTorch[Code/Formula PyTorch<br/>.pt model]
    
    CodeFormulaZDLCPred --> CodeFormulaOutput[Code/Formula Output<br/>- Code Blocks<br/>- Math Formulas]
    CodeFormulaPyTorch --> CodeFormulaOutput
    
    IOCROutput --> PageWorkerFusion[Page Worker<br/>Fusion & Processing]
    LayoutOutput --> PageWorkerFusion
    TableOutput --> PageWorkerFusion
    FigureOutput --> PageWorkerFusion
    CodeFormulaOutput --> PageWorkerFusion
    
    PageWorkerFusion --> ReadingOrder[Reading Order<br/>reading_order_rb.py]
    
    ReadingOrder --> ListNormalizer[List Item Normalizer<br/>list_marker_processor.py]
    
    ListNormalizer --> BatchComplete[Batch Complete<br/>Store Results]
    
    BatchComplete --> ResultQueue{All Batches<br/>Complete?}
    
    ResultQueue -->|No| PageServer
    ResultQueue -->|Yes| ResultWorker[Result Worker<br/>Aggregate All Batches]
    
    ResultWorker --> Serialization[Serialization<br/>Convert to DoclingDocument]
    
    Serialization --> OutputFormat{Output Format}
    
    OutputFormat -->|JSON| JSONOutput[JSON Output<br/>iocr_and_tables.json]
    OutputFormat -->|Markdown| MDOutput[Markdown Output]
    OutputFormat -->|HTML| HTMLOutput[HTML Output]
    OutputFormat -->|DoclingCore| DoclingOutput[DoclingDocument]
    
    JSONOutput --> Storage[Storage Provider<br/>S3/Local/Cloud]
    MDOutput --> Storage
    HTMLOutput --> Storage
    DoclingOutput --> Storage
    
    Storage --> Response[API Response<br/>Return to Client]
    
    Response --> End([Processed Document])
    
    %% Separate AI Worker Path for LLM tasks
    JobQueue --> AIQueue{Job Type?}
    AIQueue -->|LLM/Semantic KVP| AIWorker[AI Worker<br/>LLM Processing Only]
    AIWorker --> LLMModels[LLM Models<br/>Semantic KVP<br/>Generic KVP]
    LLMModels --> AIOutput[AI Output<br/>Key-Value Pairs<br/>Classifications]
    AIOutput --> Storage
    
    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style WDU fill:#fff3e0
    style PageWorker fill:#fff9c4,stroke:#F57F17,stroke-width:3px
    style IOCR fill:#f3e5f5
    style DoclingLayout fill:#e8f5e9
    style DoclingTable fill:#e8f5e9
    style LayoutZDLCPred fill:#ffeb3b
    style TableZDLCPred fill:#ffeb3b
    style FigureZDLCPred fill:#ffeb3b
    style CodeFormulaZDLCPred fill:#ffeb3b
    style PageWorkerFusion fill:#ffe0b2
    style AIWorker fill:#e1bee7
    style Storage fill:#e0f2f1
```

## Key Corrections

### Worker Responsibilities

#### Page Worker (Core Document Processing)
- **Loads ALL models**: OCR, Layout, Tables, Figures, Code/Formula
- **Processes document pages** with all AI models
- **Performs fusion** of all model outputs
- **Handles reading order** and list normalization
- **Stores batch results**

Code evidence:
```python
# page_worker.py line 68-69
ModelsProvider.setup(self.library_config)
ModelsProvider.load_all()  # Loads ALL models
```

#### AI Worker (LLM Processing Only)
- **Separate worker** for LLM/semantic tasks
- **Only loads LLM models**: Semantic KVP, Generic KVP
- **Handles**: Schema creation, classification, key-value extraction
- **Does NOT** handle OCR, layout, tables, or figures

Code evidence:
```python
# ai_worker.py line 91-96
ModelsProvider.get_semantic_kvp_model()
ModelsProvider.get_generic_kvp_model()
# Only LLM models, not OCR/layout/tables
```

### Processing Flow

1. **Document Input** → API Gateway → Initial Processor
2. **Initial Processor** validates and creates page batches
3. **Page Server** monitors queue and dispatches to Page Workers
4. **Page Worker**:
   - Loads ALL models (IOCR + Docling models)
   - Processes pages with OCR, layout, tables, figures, code detection
   - Fuses all outputs
   - Applies reading order and list normalization
   - Stores batch results
5. **Result Worker** (when all batches complete):
   - Aggregates all batch results
   - Serializes to final format
   - Stores in storage provider
6. **AI Worker** (separate path for LLM tasks):
   - Only for semantic KVP, classification, schema tasks
   - Does NOT process OCR/layout/tables

### ZDLC Integration

Your ZDLC predictors integrate in the **Page Worker** at these points:
- Layout Predictor ZDLC
- TableFormer ZDLC
- Figure Classifier ZDLC
- Code/Formula ZDLC

All run within the Page Worker's model loading and processing pipeline.

## Component Details

### Page Worker Architecture
```
Page Worker
├── ModelsProvider.load_all()
│   ├── IOCR Models (hrl_ocr)
│   │   ├── Text Detection
│   │   └── OCR Recognition
│   └── Docling Models (docling-ibm-models)
│       ├── Layout Predictor (PyTorch or ZDLC)
│       ├── TableFormer (PyTorch or ZDLC)
│       ├── Figure Classifier (PyTorch or ZDLC)
│       └── Code/Formula Detector (PyTorch or ZDLC)
├── Process Pages
├── Fusion Layer
├── Reading Order
├── List Normalization
└── Store Batch Results
```

### AI Worker Architecture (Separate)
```
AI Worker
├── ModelsProvider (LLM only)
│   ├── Semantic KVP Model
│   └── Generic KVP Model
├── Process LLM Tasks
│   ├── Schema Creation
│   ├── Classification
│   └── Key-Value Extraction
└── Store Results