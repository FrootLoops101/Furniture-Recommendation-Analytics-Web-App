from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict, Any
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextGeneratorService:
    """Service for generating creative product descriptions using GenAI"""
    
    def __init__(self):
        """Initialize the text generation model"""
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the text generation model"""
        try:
            logger.info(f"Loading model: {settings.GENAI_MODEL}")
            self.tokenizer = AutoTokenizer.from_pretrained(settings.GENAI_MODEL)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(settings.GENAI_MODEL)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def generate_product_description(
        self, 
        product: Dict[str, Any],
        style: str = "creative"
    ) -> str:
        """Generate a creative product description"""
        try:
            # Create prompt based on product details
            title = product.get('title', 'Product')
            material = product.get('material', 'quality materials')
            color = product.get('color', 'beautiful color')
            category = product.get('category', 'furniture')
            
            if style == "creative":
                prompt = f\"\"\"Write a creative and engaging product description for this furniture item:
Product: {title}
Material: {material}
Color: {color}
Category: {category}

Description:\"\"\"
            else:
                prompt = f\"\"\"Write a professional product description for:
{title} made of {material} in {color} color. Category: {category}

Description:\"\"\"
            
            # Generate text
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    min_length=50,
                    num_beams=4,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated description for: {title[:50]}...")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            # Fallback to template-based generation
            return self._template_description(product)
    
    def _template_description(self, product: Dict[str, Any]) -> str:
        """Fallback template-based description generation"""
        title = product.get('title', 'This product')
        material = product.get('material', 'quality materials')
        color = product.get('color', 'attractive color')
        category = product.get('category', 'furniture')
        
        templates = [
            f"Discover {title}, expertly crafted from {material} in a stunning {color} finish. Perfect for any {category} setup, this piece combines functionality with elegant design.",
            f"Elevate your space with {title}. Made from durable {material} and finished in {color}, this {category} item brings both style and practicality to your home.",
            f"Transform your interior with {title}. Featuring {material} construction and {color} aesthetics, this {category} piece is designed to impress and endure."
        ]
        
        import random
        return random.choice(templates)
    
    def generate_conversational_response(
        self, 
        query: str, 
        products: list
    ) -> str:
        """Generate a conversational response for the chat interface"""
        try:
            if not products:
                return "I couldn't find any products matching your request. Could you try rephrasing or provide more details?"
            
            product_titles = ", ".join([p.get('title', '')[:30] for p in products[:3]])
            
            prompt = f\"\"\"User asked: {query}
Found products: {product_titles}

Write a friendly, helpful response recommending these products:\"\"\"
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=256,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,
                    num_beams=3,
                    temperature=0.7
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"Error generating conversational response: {e}")
            return f"I found {len(products)} great options for you! Check them out below."

# Global instance
text_generator = TextGeneratorService()
