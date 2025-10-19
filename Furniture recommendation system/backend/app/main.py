from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import json
import logging

from config import settings
from vector_db_service import vector_db
from text_generator_service import text_generator
from schemas_product import (
    Product, 
    RecommendationRequest, 
    RecommendationResponse,
    SearchRequest
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Furniture Recommendation API",
    description="AI-powered furniture product recommendation system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
is_initialized = False
products_df = None
analytics_data = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global is_initialized, products_df, analytics_data
    
    try:
        logger.info("Starting application initialization...")
        
        # Load products data
        logger.info("Loading products data...")
        products_df = pd.read_csv('cleaned_products.csv')
        logger.info(f"Loaded {len(products_df)} products")
        
        # Load analytics data
        logger.info("Loading analytics data...")
        with open('analytics_data.json', 'r') as f:
            analytics_data = json.load(f)
        logger.info("Analytics data loaded")
        
        # Connect to vector database
        logger.info("Connecting to vector database...")
        if vector_db.connect():
            logger.info("Vector database connected")
            
            # Check if we need to upload products
            stats = vector_db.get_stats()
            if stats.get('total_vectors', 0) == 0:
                logger.info("Uploading products to vector database...")
                vector_db.upsert_products(products_df)
                logger.info("Products uploaded successfully")
        else:
            logger.warning("Could not connect to vector database. Search will not work.")
        
        # Load text generation model
        logger.info("Loading text generation model...")
        text_generator.load_model()
        logger.info("Text generation model loaded")
        
        is_initialized = True
        logger.info("✅ Application initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        logger.warning("Application started but some features may not work")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Furniture Recommendation API",
        "status": "running",
        "initialized": is_initialized
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "initialized": is_initialized,
        "products_loaded": products_df is not None,
        "analytics_loaded": analytics_data is not None,
        "vector_db_connected": vector_db.index is not None
    }

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get product recommendations based on user query"""
    try:
        if not is_initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        logger.info(f"Recommendation request: {request.query}")
        
        # Search for similar products
        filter_dict = None
        if request.filters:
            filter_dict = {}
            if 'category' in request.filters:
                filter_dict['category'] = {"$eq": request.filters['category']}
            if 'min_price' in request.filters and 'max_price' in request.filters:
                filter_dict['price'] = {
                    "$gte": request.filters['min_price'],
                    "$lte": request.filters['max_price']
                }
        
        results = vector_db.search(
            query=request.query,
            top_k=request.top_k,
            filter_dict=filter_dict
        )
        
        if not results:
            logger.warning(f"No results found for query: {request.query}")
            return RecommendationResponse(
                query=request.query,
                recommendations=[],
                generated_description="I couldn't find products matching your request. Try a different search!"
            )
        
        # Convert to Product objects
        recommendations = [
            Product(
                id=r['id'],
                title=r['title'],
                description=r['description'],
                price=r['price'] if r['price'] > 0 else None,
                category=r['category'],
                brand=r['brand'],
                material=r.get('material'),
                color=r.get('color'),
                image_url=r['image_url'],
                similarity_score=r['similarity_score']
            )
            for r in results
        ]
        
        # Generate conversational response
        generated_response = text_generator.generate_conversational_response(
            request.query, 
            results
        )
        
        logger.info(f"Returning {len(recommendations)} recommendations")
        
        return RecommendationResponse(
            query=request.query,
            recommendations=recommendations,
            generated_description=generated_response
        )
        
    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_products(request: SearchRequest):
    """Search products with filters"""
    try:
        if not is_initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")
        
        logger.info(f"Search request: {request.query}")
        
        # Build filter dictionary
        filter_dict = {}
        if request.category:
            filter_dict['category'] = {"$eq": request.category}
        if request.min_price is not None and request.max_price is not None:
            filter_dict['price'] = {
                "$gte": request.min_price,
                "$lte": request.max_price
            }
        
        results = vector_db.search(
            query=request.query,
            top_k=request.top_k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in search_products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics():
    """Get analytics data for dashboard"""
    try:
        if not is_initialized or analytics_data is None:
            raise HTTPException(status_code=503, detail="Analytics data not available")
        
        logger.info("Analytics request received")
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error in get_analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get detailed product information"""
    try:
        if not is_initialized or products_df is None:
            raise HTTPException(status_code=503, detail="Products data not available")
        
        product = products_df[products_df['uniq_id'] == product_id]
        
        if product.empty:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product_data = product.iloc[0].to_dict()
        
        # Generate creative description
        generated_desc = text_generator.generate_product_description(product_data)
        
        return {
            "id": product_id,
            "title": product_data['title'],
            "description": product_data['description'],
            "generated_description": generated_desc,
            "price": product_data['price_float'] if pd.notna(product_data['price_float']) else None,
            "category": product_data['primary_category'],
            "brand": product_data['brand'],
            "material": product_data['material'] if pd.notna(product_data['material']) else None,
            "color": product_data['color'] if pd.notna(product_data['color']) else None,
            "image_url": product_data['primary_image']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_product: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get list of all product categories"""
    try:
        if not is_initialized or products_df is None:
            raise HTTPException(status_code=503, detail="Products data not available")
        
        categories = products_df['primary_category'].unique().tolist()
        return {"categories": sorted(categories)}
        
    except Exception as e:
        logger.error(f"Error in get_categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        stats = {
            "total_products": len(products_df) if products_df is not None else 0,
            "vector_db_stats": vector_db.get_stats() if vector_db.index else {}
        }
        return stats
        
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.BACKEND_PORT)
