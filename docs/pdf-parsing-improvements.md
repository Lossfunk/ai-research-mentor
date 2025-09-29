# PDF Parsing Improvements

## Context
User feedback identified that the PDF summarization was "directionally correct but incomplete", missing:
- Sub-experiment structure and variants
- Evaluation dimensions and metrics
- Explicit findings/verdicts
- Numerical results and language-specific details
- Important caveats and limitations

## Implemented Improvements (Tier 1.5)

### 1. Enhanced Prompt Engineering
- **More detailed instructions**: Ask explicitly for sub-experiments, verdicts, numerical results, caveats
- **Structured format**: Provide clear template with Setup/Key finding/Details breakdown
- **Example-driven**: Include concrete example of desired output format

### 2. Increased Sampling Coverage
- Pages: 20 → 30 (50% increase)
- Chars per page: 1,500 → 2,000 (33% increase)  
- Total token limit: 8,000 → 12,000 (50% increase)
- **Impact**: Better coverage of document content, especially for longer papers

### 3. Explicit Structure Preservation
- Request sub-experiment variants (e.g., "1a: score-based", "1b: ranking-based")
- Ask for evaluation dimensions/metrics
- Require explicit findings and verdicts ("what was learned?")
- Request numerical results when mentioned
- Ask for caveats and limitations

## Potential Tier 2 Enhancements

If current improvements still don't meet quality requirements, consider:

### 1. Multi-Pass Extraction
**Approach**: 
- Pass 1: Extract document structure (sections, subsections)
- Pass 2: Extract experiments from relevant sections
- Pass 3: Extract numerical results from tables/figures

**Benefits**: More accurate, preserves hierarchical structure
**Cost**: 3x LLM calls, slower processing

### 2. Enhanced PDF Parser
**Current**: PyPDF (basic text extraction)
**Alternatives**: 
- `pdfplumber`: Better table and layout handling
- `unstructured`: ML-based document understanding
- `pymupdf`: Better text extraction quality

**Benefits**: Better handling of tables, multi-column layouts, figures
**Cost**: Additional dependencies, more complex implementation

### 3. Semantic Chunking
**Current**: Fixed 1200-char chunks with 180 overlap
**Improvement**: 
- Respect paragraph boundaries
- Keep experimental descriptions together
- Preserve section headers with content

**Benefits**: More coherent chunks, better retrieval
**Cost**: More complex chunking logic

### 4. Hierarchical Retrieval
**Approach**:
- Level 1: Section-level search (which sections are relevant?)
- Level 2: Chunk-level search within relevant sections

**Benefits**: Better relevance, less noise
**Cost**: More complex retrieval logic, potentially slower

### 5. Structured Data Extraction
**Approach**:
- Use LLM to extract structured data (JSON format)
- Parse experiments, methods, results into typed objects
- Store in queryable format

**Benefits**: Precise querying, better context injection
**Cost**: More complex implementation, schema management

## Testing Strategy

To evaluate improvements:
1. Test with the multilingual research paper
2. Compare extraction against ground truth
3. Check for:
   - Sub-experiment variants captured
   - Evaluation dimensions listed
   - Explicit verdicts present
   - Numerical results included
   - Caveats mentioned
4. Iterate if gaps remain

## Recommendation

**Current approach**: Test Tier 1.5 improvements first
- Lower cost (just prompt engineering + sampling)
- Should address most feedback issues
- Easy to iterate

**If still insufficient**: Implement Tier 2 in order:
1. Multi-pass extraction (biggest quality gain)
2. Enhanced PDF parser (better base extraction)
3. Semantic chunking (better retrieval)
4. Hierarchical retrieval (optional refinement)
5. Structured extraction (if precision critical)
