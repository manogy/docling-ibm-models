# Watson Document Understanding - End-to-End Document Processing Flow

## Mermaid Diagram

```mermaid
graph TB
    Start([Document Input<br/>PDF/Image/DOCX]) --> WDU[Watson Document Understanding Service]
    
    WDU --> APIGateway[API Gateway<br/>REST/gRPC]
    APIGateway --> JobQueue[Job Queue]
    
    JobQueue --> PageWorker[Page Worker<br/>Document Segmentation]
    PageWorker --> Pages[Split into Pages]
    
    Pages --> AIWorker[AI Worker<br/>Model Orchestration]
    
    AIWorker --> ModelProvider[Models Provider<br/>Load & Manage Models]
    
    ModelProvider --> OCRPath{OCR Path}
    ModelProvider --> LayoutPath{Layout Path}
    ModelProvider --> TablePath{Table Path}
    
    OCRPath --> IOCR[IOCR Models<br/>hrl_ocr package]
    IOCR --> TextDetection[Text Detection Model]
    IOCR --> OCRRecognition[OCR Recognition Model]
    
    TextDetection --> IOCROutput[IOCR Output<br/>- Text Lines<br/>- Bounding Boxes<br/>- Blocks]
    OCRRecognition --> IOCROutput
    
    LayoutPath --> DoclingLayout[Docling Layout Model<br/>docling-ibm-models]
    DoclingLayout --> LayoutPredictor[Layout Predictor<br/>layout_predictor.py]
    LayoutPredictor --> LayoutZDLC{Use ZDLC?}
    
    LayoutZDLC -->|Yes| LayoutZDLCPred[Layout Predictor ZDLC<br/>layout_predictor_zdlc.py<br/>Uses .so compiled model]
    LayoutZDLC -->|No| LayoutPyTorch[Layout Predictor PyTorch<br/>Uses .pt model]
    
    LayoutZDLCPred --> LayoutOutput[Layout Output<br/>- Page Elements<br/>- Sections<br/>- Headers/Footers]
    LayoutPyTorch --> LayoutOutput
    
    TablePath --> DoclingTable[Docling Table Models<br/>docling-ibm-models]
    DoclingTable --> TableFormer[TableFormer<br/>tf_predictor.py]
    TableFormer --> TableZDLC{Use ZDLC?}
    
    TableZDLC -->|Yes| TableZDLCPred[TableFormer ZDLC<br/>tf_predictor_zdlc.py<br/>Uses .so compiled model]
    TableZDLC -->|No| TablePyTorch[TableFormer PyTorch<br/>Uses .pt model]
    
    TableZDLCPred --> TableOutput[Table Output<br/>- Table Structure<br/>- Cells<br/>- Rows/Columns]
    TablePyTorch --> TableOutput
    
    DoclingLayout --> FigureClassifier[Figure Classifier<br/>document_figure_classifier]
    FigureClassifier --> FigureZDLC{Use ZDLC?}
    
    FigureZDLC -->|Yes| FigureZDLCPred[Figure Classifier ZDLC<br/>Uses .so compiled model]
    FigureZDLC -->|No| FigurePyTorch[Figure Classifier PyTorch<br/>Uses .pt model]
    
    FigureZDLCPred --> FigureOutput[Figure Classification<br/>- Chart Types<br/>- Diagrams<br/>- Images]
    FigurePyTorch --> FigureOutput
    
    DoclingLayout --> CodeFormula[Code/Formula Detector<br/>code_formula_predictor]
    CodeFormula --> CodeFormulaZDLC{Use ZDLC?}
    
    CodeFormulaZDLC -->|Yes| CodeFormulaZDLCPred[Code/Formula ZDLC<br/>Uses .so compiled model]
    CodeFormulaZDLC -->|No| CodeFormulaPyTorch[Code/Formula PyTorch<br/>Uses .pt model]
    
    CodeFormulaZDLCPred --> CodeFormulaOutput[Code/Formula Output<br/>- Code Blocks<br/>- Math Formulas]
    CodeFormulaPyTorch --> CodeFormulaOutput
    
    IOCROutput --> Fusion[Fusion Layer<br/>parse_table_fusion]
    LayoutOutput --> Fusion
    TableOutput --> Fusion
    FigureOutput --> Fusion
    CodeFormulaOutput --> Fusion
    
    Fusion --> ReadingOrder[Reading Order<br/>reading_order_rb.py<br/>Determine logical flow]
    
    ReadingOrder --> ListNormalizer[List Item Normalizer<br/>list_marker_processor.py<br/>Process bullets/numbering]
    
    ListNormalizer --> Serialization[Serialization<br/>Convert to DoclingDocument]
    
    Serialization --> OutputFormat{Output Format}
    
    OutputFormat -->|JSON| JSONOutput[JSON Output<br/>iocr_and_tables.json]
    OutputFormat -->|Markdown| MDOutput[Markdown Output]
    OutputFormat -->|HTML| HTMLOutput[HTML Output]
    OutputFormat -->|DoclingCore| DoclingOutput[DoclingDocument<br/>Structured Format]
    
    JSONOutput --> Storage[Storage Provider<br/>S3/Local/Cloud]
    MDOutput --> Storage
    HTMLOutput --> Storage
    DoclingOutput --> Storage
    
    Storage --> ResultWorker[Result Worker<br/>Package Results]
    
    ResultWorker --> Response[API Response<br/>Return to Client]
    
    Response --> End([Processed Document<br/>with Extracted Content])
    
    style Start fill:#e1f5ff
    style End fill:#c8e6c9
    style WDU fill:#fff3e0
    style IOCR fill:#f3e5f5
    style DoclingLayout fill:#e8f5e9
    style DoclingTable fill:#e8f5e9
    style LayoutZDLCPred fill:#ffeb3b
    style TableZDLCPred fill:#ffeb3b
    style FigureZDLCPred fill:#ffeb3b
    style CodeFormulaZDLCPred fill:#ffeb3b
    style Fusion fill:#ffe0b2
    style Storage fill:#e0f2f1
```

## Component Details

### 1. **Input Layer**
- Accepts: PDF, Images (PNG, JPG), DOCX
- Entry point: REST API or gRPC endpoint

### 2. **WDU Service Layer**
- **API Gateway**: Routes requests to appropriate workers
- **Job Queue**: Manages processing queue
- **Page Worker**: Splits documents into pages

### 3. **AI Worker & Model Provider**
- **AI Worker**: Orchestrates model execution
- **Models Provider**: Manages model loading and lifecycle
- Supports both PyTorch and ZDLC backends

### 4. **IOCR Processing** (hrl_ocr package)
- **Text Detection**: Locates text regions
- **OCR Recognition**: Extracts text content
- **Output**: Text lines, bounding boxes, blocks

### 5. **Docling Models Processing** (docling-ibm-models)

#### Layout Analysis
- **Model**: Layout Predictor
- **Backends**: PyTorch (.pt) or ZDLC (.so)
- **Output**: Page structure, sections, headers

#### Table Extraction
- **Model**: TableFormer
- **Backends**: PyTorch (.pt) or ZDLC (.so)
- **Output**: Table structure, cells, rows/columns

#### Figure Classification
- **Model**: Document Figure Classifier
- **Backends**: PyTorch (.pt) or ZDLC (.so)
- **Output**: Chart types, diagrams, image classification

#### Code/Formula Detection
- **Model**: Code Formula Predictor
- **Backends**: PyTorch (.pt) or ZDLC (.so)
- **Output**: Code blocks, mathematical formulas

### 6. **Post-Processing**
- **Fusion Layer**: Combines IOCR + Docling outputs
- **Reading Order**: Determines logical document flow
- **List Normalizer**: Processes bullets and numbering

### 7. **Serialization & Output**
- **Formats**: JSON, Markdown, HTML, DoclingDocument
- **Storage**: S3, Local filesystem, Cloud storage
- **Response**: Packaged results returned to client

## ZDLC Integration Points

The ZDLC predictors you created integrate at these decision points:

1. **Layout Predictor**: `layout_predictor_zdlc.py`
2. **TableFormer**: `tf_predictor_zdlc.py`
3. **Figure Classifier**: `document_figure_classifier_predictor_zdlc.py`
4. **Code/Formula**: `code_formula_predictor_zdlc.py`

Each ZDLC predictor:
- Uses compiled `.so` models instead of `.pt` PyTorch models
- Provides same API interface as PyTorch version
- Offers faster inference on IBM Z systems
- Maintains identical output data types

## Performance Benefits with ZDLC

- **Faster Inference**: 2-5x speedup on IBM Z
- **Lower Memory**: Optimized memory usage
- **Better CPU Utilization**: Leverages Z architecture
- **Production Ready**: Drop-in replacement for PyTorch