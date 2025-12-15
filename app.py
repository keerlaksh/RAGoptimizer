# streamlit_app.py
"""
Streamlit Interface for RAG Experimental Framework

This interface allows users to:
1. Upload PDFs and configure API keys directly in the UI
2. Run controlled experiments across multiple RAG configurations
3. Compare pipeline performance scientifically
4. Get answers using the best-performing pipeline
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import tempfile
from pathlib import Path

# ============================================
# DYNAMIC IMPORTS (with API key injection)
# ============================================

def initialize_rag_system(api_key=None):
    """Initialize RAG system with provided API key"""
    if api_key:
        os.environ['GROQ_API_KEY'] = api_key
    
    # Import after setting API key
    from rag_experimental_framework import (
        EXPERIMENTAL_PIPELINES,
        EMBEDDERS,
        DBS,
        load_and_chunk_folder,
        retrieve,
        generate_answer
    )
    from rag_evaluation_metrics import evaluate_pipeline, compare_pipelines
    
    return {
        'pipelines': EXPERIMENTAL_PIPELINES,
        'embedders': EMBEDDERS,
        'dbs': DBS,
        'load_and_chunk_folder': load_and_chunk_folder,
        'retrieve': retrieve,
        'generate_answer': generate_answer,
        'evaluate_pipeline': evaluate_pipeline,
        'compare_pipelines': compare_pipelines
    }

# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="RAG Experimental Framework",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

if 'indexed' not in st.session_state:
    st.session_state.indexed = False
if 'experiment_results' not in st.session_state:
    st.session_state.experiment_results = None
if 'best_pipeline' not in st.session_state:
    st.session_state.best_pipeline = None
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

# ============================================
# HEADER
# ============================================

st.title("🔬 RAG Experimental Framework")
st.markdown("""
### A Controlled Experimental System for RAG Pipeline Optimization

This system treats RAG design as a **scientific experiment**, measuring:
- 📊 **Information Flow Integrity**: Does retrieval fetch useful information?
- 🔗 **Retrieval-Generation Coupling**: Does better retrieval lead to better answers?
- ⚖️ **Hallucination vs Coverage Trade-off**: Groundedness vs completeness
""")

st.divider()

# ============================================
# SIDEBAR - SETUP & CONFIGURATION
# ============================================

with st.sidebar:
    st.header("⚙️ System Setup")
    
    # ----------------------------------------
    # 1. API KEY CONFIGURATION
    # ----------------------------------------
    st.subheader("1. Configure LLM API")
    
    with st.expander("🔑 API Key Setup", expanded=not st.session_state.rag_system):
        st.markdown("""
        **Get a FREE Groq API Key:**
        1. Visit [console.groq.com](https://console.groq.com/)
        2. Sign up (free)
        3. Create an API key
        4. Paste it below
        """)
        
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Your API key is stored only in this session and never saved"
        )
        
        if api_key and api_key.startswith('gsk_'):
            if st.button("✅ Initialize System with API Key", use_container_width=True):
                with st.spinner("Initializing RAG system..."):
                    try:
                        st.session_state.rag_system = initialize_rag_system(api_key)
                        st.success("✅ System initialized with LLM!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        elif api_key:
            st.warning("⚠️ API key should start with 'gsk_'")
        
        st.markdown("---")
        st.markdown("**Don't have an API key?**")
        st.info("💡 The system will work without it, but will show retrieved context instead of generated answers.")
        
        if st.button("Continue Without API Key", use_container_width=True):
            with st.spinner("Initializing RAG system..."):
                try:
                    st.session_state.rag_system = initialize_rag_system(None)
                    st.info("ℹ️ System initialized (no LLM - will show context)")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    if st.session_state.rag_system:
        st.success("✅ System Ready")
    
    st.divider()
    
    # ----------------------------------------
    # 2. PDF UPLOAD
    # ----------------------------------------
    st.subheader("2. Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents for analysis"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        with st.expander("📄 View Uploaded Files"):
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")
        
        # Index button
        if st.button("🚀 Index Documents", use_container_width=True, type="primary"):
            if not st.session_state.rag_system:
                st.error("⚠️ Please initialize the system first (Step 1)")
            else:
                with st.spinner("Indexing documents across all pipelines..."):
                    try:
                        # Create temporary directory for uploaded files
                        if st.session_state.temp_dir is None:
                            st.session_state.temp_dir = tempfile.mkdtemp()
                        
                        temp_dir = st.session_state.temp_dir
                        
                        # Save uploaded files
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                        
                        # Index documents
                        rag = st.session_state.rag_system
                        
                        for name, cfg in rag['pipelines'].items():
                            st.write(f"Indexing pipeline: {name}...")
                            
                            chunks, metas = rag['load_and_chunk_folder'](
                                temp_dir,
                                cfg['chunk_size'],
                                cfg['chunk_strategy']
                            )
                            
                            if len(chunks) == 0:
                                st.warning(f"No chunks created for {name}")
                                continue
                            
                            embeddings = rag['embedders'][name].encode(
                                chunks,
                                show_progress_bar=True
                            )
                            embeddings = [e.tolist() for e in embeddings]
                            
                            ids = [f"{name}_{i}" for i in range(len(chunks))]
                            
                            try:
                                rag['dbs'][name].add(
                                    ids=ids,
                                    documents=chunks,
                                    embeddings=embeddings,
                                    metadatas=metas,
                                )
                                st.write(f"✅ {name}: {len(chunks)} chunks indexed")
                            except Exception as e:
                                st.write(f"ℹ️ {name}: Already indexed or error")
                        
                        st.session_state.indexed = True
                        st.session_state.uploaded_files = uploaded_files
                        st.success("✅ All documents indexed!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error during indexing: {e}")
    
    if st.session_state.indexed:
        st.success(f"✅ {len(st.session_state.uploaded_files)} document(s) indexed")
    
    st.divider()
    
    # ----------------------------------------
    # 3. EXPERIMENTAL PARAMETERS
    # ----------------------------------------
    st.subheader("3. Experimental Parameters")
    
    top_k = st.slider(
        "Top-K Documents to Retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of chunks to retrieve per query"
    )
    
    st.divider()
    
    # ----------------------------------------
    # 4. PIPELINE INFORMATION
    # ----------------------------------------
    if st.session_state.rag_system:
        st.subheader("4. Experimental Configurations")
        
        pipelines = st.session_state.rag_system['pipelines']
        st.write(f"**Testing {len(pipelines)} pipelines:**")
        
        for name, cfg in pipelines.items():
            with st.expander(f"📋 {name}"):
                st.write(f"**Model:** {cfg['model']}")
                st.write(f"**Chunk Size:** {cfg['chunk_size']}")
                st.write(f"**Strategy:** {cfg['chunk_strategy']}")
                st.write(f"**Description:** {cfg['description']}")

# ============================================
# MAIN INTERFACE - EXPERIMENT RUNNER
# ============================================

if not st.session_state.rag_system:
    st.info("👈 Please initialize the system in the sidebar (Step 1)")
    st.stop()

if not st.session_state.indexed:
    st.info("👈 Please upload and index documents in the sidebar (Step 2)")
    st.stop()

st.header("🧪 Run Experiment")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Enter your query:",
        placeholder="Type your query here",
        help="This query will be tested across all RAG configurations"
    )

with col2:
    run_button = st.button("🔬 Run Experiment", use_container_width=True, type="primary")

if run_button and query:
    with st.spinner("Running controlled experiment across all pipelines..."):
        try:
            rag = st.session_state.rag_system
            
            # Run experiment across all pipelines
            results = {}
            
            for pipeline_name, cfg in rag['pipelines'].items():
                # Retrieve documents
                retrieved_docs, metas = rag['retrieve'](pipeline_name, query, top_k)
                
                # Generate answer
                answer = rag['generate_answer'](query, retrieved_docs)
                
                results[pipeline_name] = {
                    "config": cfg,
                    "retrieved_docs": retrieved_docs,
                    "metadata": metas,
                    "answer": answer
                }
            
            # Evaluate each pipeline
            pipeline_metrics = {}
            
            for pipeline_name, result in results.items():
                metrics = rag['evaluate_pipeline'](
                    queries=[query],
                    responses=[result["answer"]],
                    retrieved_docs=[result["retrieved_docs"]],
                    top_k=top_k
                )
                pipeline_metrics[pipeline_name] = metrics
            
            # Compare and rank
            ranked_pipelines, comparison = rag['compare_pipelines'](pipeline_metrics)
            
            # Store results
            st.session_state.experiment_results = {
                "query": query,
                "results": results,
                "metrics": pipeline_metrics,
                "ranked": ranked_pipelines,
                "comparison": comparison
            }
            
            st.session_state.best_pipeline = ranked_pipelines[0][0]
            st.success("✅ Experiment complete!")
            
        except Exception as e:
            st.error(f"❌ Error during experiment: {e}")
            st.exception(e)

# ============================================
# RESULTS VISUALIZATION
# ============================================

if st.session_state.experiment_results:
    st.divider()
    st.header("📊 Experimental Results")
    
    results = st.session_state.experiment_results
    best_pipeline = st.session_state.best_pipeline
    
    # ----------------------------------------
    # 1. BEST PIPELINE ANSWER
    # ----------------------------------------
    
    st.subheader("🏆 Best Pipeline Answer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**Selected Pipeline:** {best_pipeline}")
        best_config = results["results"][best_pipeline]["config"]
        st.caption(f"📊 Using {best_config['model']} embeddings with {best_config['chunk_size']}-word chunks")
        st.caption("✅ This answer was generated using the best pipeline's retrieved documents")
    
    with col2:
        st.metric("Chunk Size", f"{best_config['chunk_size']} words")
        st.metric("Embedding Model", best_config['model'].split('/')[-1][:15] + "...")
    
    best_answer = results["results"][best_pipeline]["answer"]
    best_retrieved_docs = results["results"][best_pipeline]["retrieved_docs"]
    
    st.markdown("### Answer:")
    with st.expander("ℹ️ About this answer", expanded=False):
        st.markdown(f"""
        **How this answer was generated:**
        1. The **{best_pipeline}** pipeline retrieved {len(best_retrieved_docs)} document chunks
        2. The LLM generated this answer using **only** those retrieved chunks
        3. This pipeline was selected as best based on:
           - Retrieval quality (how relevant the chunks are)
           - Answer groundedness (how well the answer uses the chunks)
           - Context utilization (how much of the retrieved info is used)
           - Low hallucination rate
        
        **This ensures the answer is based on the best-performing retrieval pipeline.**
        """)
    st.markdown(best_answer)
    
    # Show why it was selected
    best_metrics = results["metrics"][best_pipeline]
    
    st.markdown("### Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Overall Score",
            f"{best_metrics['overall_score']:.3f}",
            help="Composite score based on all metrics"
        )
    with col2:
        st.metric(
            "Groundedness",
            f"{best_metrics['faithfulness']['Groundedness']:.3f}",
            help="% of answer supported by sources"
        )
    with col3:
        st.metric(
            "Context Use",
            f"{best_metrics['coupling']['Context Utilization']:.3f}",
            help="How well retrieval was utilized"
        )
    with col4:
        st.metric(
            "Hallucination Rate",
            f"{best_metrics['faithfulness']['Hallucination Rate']:.3f}",
            delta=f"{-best_metrics['faithfulness']['Hallucination Rate']:.3f}",
            delta_color="inverse",
            help="% of answer not in sources (lower is better)"
        )
    
    st.divider()
    
    # ----------------------------------------
    # 2. COMPARATIVE ANALYSIS
    # ----------------------------------------
    
    st.subheader("📈 Pipeline Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for pipeline_name, scores in results["comparison"].items():
        comparison_data.append({
            "Pipeline": pipeline_name,
            "Overall Score": scores["Overall Score"],
            "Groundedness": scores["Groundedness"],
            "Context Use": scores["Context Use"],
            "Hallucination": scores["Hallucination"],
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("Overall Score", ascending=False)
    
    # Display table
    st.dataframe(
        df.style.background_gradient(cmap='RdYlGn', subset=['Overall Score', 'Groundedness', 'Context Use'])
               .background_gradient(cmap='RdYlGn_r', subset=['Hallucination']),
        use_container_width=True,
        hide_index=True
    )
    
    # ----------------------------------------
    # 3. VISUAL COMPARISONS
    # ----------------------------------------
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Overall Performance", 
        "🔗 Information Flow", 
        "⚖️ Hallucination vs Coverage"
    ])
    
    with tab1:
        # Overall score comparison
        fig = px.bar(
            df,
            x="Pipeline",
            y="Overall Score",
            title="Overall Pipeline Performance",
            color="Overall Score",
            color_continuous_scale="RdYlGn"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Information flow metrics
        flow_data = []
        for pipeline_name, metrics in results["metrics"].items():
            flow_data.append({
                "Pipeline": pipeline_name,
                "Retrieval Quality (nDCG)": metrics["retrieval"]["nDCG@5"],
                "Context Utilization": metrics["coupling"]["Context Utilization"],
                "Answer Relevance": metrics["coupling"]["Answer Relevance"]
            })
        
        flow_df = pd.DataFrame(flow_data)
        
        fig = go.Figure()
        
        for col in ["Retrieval Quality (nDCG)", "Context Utilization", "Answer Relevance"]:
            fig.add_trace(go.Bar(
                name=col,
                x=flow_df["Pipeline"],
                y=flow_df[col]
            ))
        
        fig.update_layout(
            title="Information Flow Integrity Analysis",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Key Insight:** This chart shows **retrieval-generation coupling** — 
        whether better retrieval actually leads to better answer quality.
        """)
    
    with tab3:
        # Hallucination vs Coverage trade-off
        tradeoff_data = []
        for pipeline_name, metrics in results["metrics"].items():
            tradeoff_data.append({
                "Pipeline": pipeline_name,
                "Groundedness": metrics["faithfulness"]["Groundedness"],
                "Hallucination": metrics["faithfulness"]["Hallucination Rate"],
                "Coverage": metrics["faithfulness"]["Coverage Score"]
            })
        
        tradeoff_df = pd.DataFrame(tradeoff_data)
        
        # Scatter plot: Groundedness vs Hallucination
        fig = px.scatter(
            tradeoff_df,
            x="Hallucination",
            y="Groundedness",
            size="Coverage",
            color="Pipeline",
            title="Hallucination vs Groundedness Trade-off",
            labels={
                "Hallucination": "Hallucination Rate (lower is better)",
                "Groundedness": "Groundedness (higher is better)"
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Key Insight:** This visualizes the fundamental trade-off in RAG systems —
        pipelines in the top-left quadrant (low hallucination, high groundedness) are optimal.
        """)
    
    st.divider()
    
    # ----------------------------------------
    # 4. DETAILED METRICS
    # ----------------------------------------
    
    with st.expander("📋 Detailed Metrics for All Pipelines"):
        for pipeline_name in results["ranked"]:
            pipeline_name = pipeline_name[0]
            st.subheader(f"{pipeline_name}")
            
            metrics = results["metrics"][pipeline_name]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Retrieval Metrics**")
                st.json(metrics["retrieval"])
                
                st.markdown("**Coupling Metrics**")
                st.json(metrics["coupling"])
            
            with col2:
                st.markdown("**Faithfulness Metrics**")
                st.json(metrics["faithfulness"])
                
                st.markdown("**Generation Quality**")
                st.json(metrics["generation"])
            
            st.markdown("---")
    
    # ----------------------------------------
    # 5. ALL PIPELINE ANSWERS
    # ----------------------------------------
    
    with st.expander("💬 View All Pipeline Answers"):
        for pipeline_name, result in results["results"].items():
            st.markdown(f"### {pipeline_name}")
            st.markdown(result['answer'])
            st.markdown("---")
    
    # ----------------------------------------
    # 6. RETRIEVED DOCUMENTS
    # ----------------------------------------
    
    with st.expander("📚 View Retrieved Documents (Best Pipeline)", expanded=False):
        st.info(f"These are the **{len(results['results'][best_pipeline]['retrieved_docs'])} documents** that the best pipeline ({best_pipeline}) retrieved and used to generate the answer above.")
        best_docs = results["results"][best_pipeline]["retrieved_docs"]
        best_metas = results["results"][best_pipeline]["metadata"]
        
        for i, (doc, meta) in enumerate(zip(best_docs, best_metas), 1):
            st.markdown(f"**Source {i}** - {meta.get('source', 'Unknown')} (Chunk #{meta.get('chunk', '?')})")
            st.text_area(f"Content {i}", doc, height=150, key=f"doc_{i}")

# ============================================
# FOOTER
# ============================================

st.divider()

with st.expander("ℹ️ About This Framework"):
    st.markdown("""
    ## What Makes This Different?
    
    This is not just a RAG system — it's a **controlled experimental framework** for RAG optimization.
    
    ### The Three Core Measurements:
    
    1. **Information Flow Integrity**
       - Does the retrieval system fetch relevant information?
       - Measured via: Hit Rate, MRR, nDCG, Precision/Recall
    
    2. **Retrieval-Generation Coupling**
       - Does better retrieval actually improve answer quality?
       - Measured via: Context Utilization, Answer Relevance
       - **This is the key insight most RAG systems miss**
    
    3. **Hallucination vs Coverage Trade-off**
       - Are answers grounded in sources?
       - Do they cover the retrieved information?
       - Measured via: Groundedness, Hallucination Rate, Coverage Score
    
    ### Why This Matters:
    
    - Most RAG systems pick arbitrary configurations and hope they work
    - This framework **proves** which configuration is best for your use case
    - It measures the **causal relationship** between retrieval quality and answer quality
    - It's a **reproducible, scientific approach** to RAG optimization
    """)