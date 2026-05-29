"""
Dutch Bureaucracy Assistant - Complete Backend
Single file FastAPI application for analyzing Dutch government letters

Installation:
1. pip install fastapi uvicorn python-dotenv httpx pytesseract pillow python-multipart groq pydantic
2. Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
3. Create .env file with: GROQ_API_KEY=your_key_here
4. Run: python this_file.py

Author: Dutch Bureaucracy Assistant
Version: 1.0.0
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import re
import json
import tempfile
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Image processing
from PIL import Image
import pytesseract

# LLM
from groq import Groq

# Environment variables
from dotenv import load_dotenv

# Run server
import uvicorn

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

load_dotenv()

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class OCRResponse(BaseModel):
    """Response model for OCR endpoint"""
    text: str
    success: bool = True
    error: Optional[str] = None


class AnalyzeRequest(BaseModel):
    """Request model for analyze endpoint"""
    text: str = Field(..., min_length=10, description="Letter content to analyze")


class Deadline(BaseModel):
    """Deadline model for extraction"""
    date: str  # Format: YYYY-MM-DD
    action: str


class AnalyzeResponse(BaseModel):
    """Response model for analyze endpoint"""
    explanation: str
    deadlines: List[Deadline]
    obligations: List[str]
    source: str


class FormFillRequest(BaseModel):
    """Request model for formfill endpoint"""
    text: str = Field(..., min_length=10, description="Letter content for form filling")


class FormField(BaseModel):
    """Individual form field model"""
    field_name: str
    field_value: str
    confidence: Optional[str] = None


class FormFillResponse(BaseModel):
    """Response model for formfill endpoint"""
    form_type: str
    fields: List[FormField]
    notes: Optional[str] = None


# ============================================================================
# UTILITY FUNCTIONS (Helpers)
# ============================================================================

def extract_dates_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Extract dates and their context from text.
    Returns list of (date_string, context_sentence)
    """
    dutch_months = {
        'januari': 1, 'februari': 2, 'maart': 3, 'april': 4, 'mei': 5, 'juni': 6,
        'juli': 7, 'augustus': 8, 'september': 9, 'oktober': 10, 'november': 11, 'december': 12
    }
    
    date_patterns = [
        (r'\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b', 'numeric'),
        (r'\b(\d{1,2})\s+(' + '|'.join(dutch_months.keys()) + r')\s+(\d{4})\b', 'dutch'),
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b', 'english')
    ]
    
    dates_with_context = []
    sentences = re.split(r'[.!?]+', text.lower())
    
    for sentence in sentences:
        for pattern, _ in date_patterns:
            matches = re.finditer(pattern, sentence, re.IGNORECASE)
            for match in matches:
                date_str = match.group(0)
                dates_with_context.append((date_str, sentence.strip()))
                break
    
    return dates_with_context


def parse_dutch_date(date_str: str, context: str) -> Optional[str]:
    """Parse a Dutch date string to YYYY-MM-DD format."""
    dutch_months = {
        'januari': '01', 'februari': '02', 'maart': '03', 'april': '04',
        'mei': '05', 'juni': '06', 'juli': '07', 'augustus': '08',
        'september': '09', 'oktober': '10', 'november': '11', 'december': '12'
    }
    
    try:
        dd_mm_yyyy = re.match(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', date_str)
        if dd_mm_yyyy:
            day, month, year = dd_mm_yyyy.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"
        
        for dutch_month, month_num in dutch_months.items():
            if dutch_month in date_str.lower():
                match = re.match(r'(\d{1,2})\s+' + dutch_month + r'\s+(\d{4})', date_str.lower())
                if match:
                    day, year = match.groups()
                    return f"{year}-{month_num}-{int(day):02d}"
        
        return None
    except Exception:
        return None


def extract_action_from_sentence(sentence: str) -> str:
    """Extract action verb from a sentence to understand what needs to be done."""
    action_keywords = {
        'betalen': 'betaal het bedrag',
        'bezwaar': 'maak bezwaar',
        'aanvragen': 'vraag aan',
        'melden': 'meld het',
        'indienen': 'dien in',
        'sturen': 'stuur op',
        'reageren': 'reageer',
        'ondertekenen': 'onderteken',
        'versturen': 'verstuur',
        'bellen': 'bel',
        'mailen': 'mail',
        'langskomen': 'kom langs'
    }
    
    sentence_lower = sentence.lower()
    for keyword, action in action_keywords.items():
        if keyword in sentence_lower:
            return action
    
    return 'neem actie'


def validate_dutch_postal_code(postal_code: str) -> bool:
    """Validate Dutch postal code format (1234 AB)"""
    pattern = r'^[1-9][0-9]{3}\s?[A-Z]{2}$'
    return bool(re.match(pattern, postal_code.strip().upper()))


def extract_bsn(bsn: str) -> Optional[str]:
    """Validate and format BSN (Dutch citizen service number)"""
    bsn_clean = re.sub(r'[\s-]', '', bsn)
    if re.match(r'^\d{8,9}$', bsn_clean):
        return bsn_clean
    return None


# ============================================================================
# OCR SERVICE
# ============================================================================

class OCRService:
    """Handles Optical Character Recognition from images"""
    
    def __init__(self):
        """Initialize OCR service with Tesseract"""
        # Check if Tesseract is installed
        try:
            pytesseract.get_tesseract_version()
            self.tesseract_available = True
        except Exception:
            self.tesseract_available = False
            print("WARNING: Tesseract not found. Install it for OCR functionality.")
    
    async def extract_text_from_image(self, image_file) -> str:
        """
        Extract text from an uploaded image file using Tesseract.
        
        Args:
            image_file: UploadFile object from FastAPI
            
        Returns:
            Extracted text as string
        """
        if not self.tesseract_available:
            raise Exception("Tesseract OCR is not installed. Please install it first.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            # Read and convert image to RGB
            image = Image.open(image_file.file)
            
            # Convert to RGB if necessary
            if image.mode in ('RGBA', 'LA', 'P'):
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'RGBA':
                    rgb_image.paste(image, mask=image.split()[-1])
                else:
                    rgb_image.paste(image)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Save the processed image
            image.save(tmp_file.name, 'PNG')
            tmp_file_path = tmp_file.name
        
        try:
            # Configure Tesseract for Dutch and English
            custom_config = r'--oem 3 --psm 6 -l nld+eng'
            text = pytesseract.image_to_string(Image.open(tmp_file_path), config=custom_config)
            return text.strip()
        except Exception as e:
            raise Exception(f"OCR failed: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)


# ============================================================================
# LLM SERVICE (Groq API)
# ============================================================================

class LLMService:
    """Handles all LLM interactions using Groq API"""
    
    def __init__(self):
        """Initialize Groq client with API key"""
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            print("WARNING: GROQ_API_KEY not found in environment variables")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
        self.model = "mixtral-8x7b-32768"
    
    async def analyze_letter(self, text: str) -> Dict[str, Any]:
        """
        Analyze a Dutch government letter using LLM.
        
        Args:
            text: The extracted letter content
            
        Returns:
            Dictionary with explanation, deadlines, obligations, and source
        """
        if not self.client:
            # Return fallback response if no API key
            return {
                "explanation": "Groq API key not configured. Please add GROQ_API_KEY to .env file.",
                "deadlines": [],
                "obligations": ["Configureer de GROQ_API_KEY in het .env bestand"],
                "source": "Unknown"
            }
        
        prompt = f"""Je bent een expert in Nederlandse overheidsbrieven. Analyseer de volgende brief en geef een gestructureerde output in JSON formaat.

Brief inhoud:
{text}

Je moet het volgende extraheren:

1. explanation: Een eenvoudige uitleg in het Nederlands voor een gewone burger. Max 3 zinnen. Leg uit wat de brief betekent en wat er moet gebeuren.

2. deadlines: Lijst van deadlines met datum (YYYY-MM-DD formaat) en actie. Alleen als er expliciet een datum genoemd wordt.

3. obligations: Lijst van verplichtingen die de burger moet uitvoeren (max 5).

4. source: Van welke overheidsinstantie komt deze brief? Kies uit: Belastingdienst, DUO, Toeslagen, Gemeente, UWV, IND, CJIB, Unknown

Output alleen JSON, geen extra tekst. Gebruik dit formaat:
{{
  "explanation": "...",
  "deadlines": [
    {{"date": "2024-12-31", "action": "Betaal de aanslag"}}
  ],
  "obligations": ["Verplichting 1", "Verplichting 2"],
  "source": "Belastingdienst"
}}

Let op: Wees duidelijk, gebruik eenvoudige woorden, en vermeld alleen deadlines als de datum expliciet in de brief staat."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Je bent een behulpzame assistent die overheidsbrieven uitlegt aan gewone burgers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['explanation', 'deadlines', 'obligations', 'source']
            for field in required_fields:
                if field not in result:
                    result[field] = [] if field in ['deadlines', 'obligations'] else "Onbekend"
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "explanation": "Kon de brief niet volledig analyseren. Probeer het opnieuw of raadpleeg een hulpverlener.",
                "deadlines": [],
                "obligations": ["Lees de brief zorgvuldig door", "Neem contact op met de afzender bij vragen"],
                "source": "Unknown"
            }
        except Exception as e:
            print(f"LLM API error: {e}")
            raise Exception(f"Failed to analyze letter: {str(e)}")
    
    async def extract_form_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract form fields from a government letter for auto-filling.
        
        Args:
            text: The extracted letter content
            
        Returns:
            Dictionary with form type and extracted fields
        """
        if not self.client:
            return {
                "form_type": "onbekend",
                "fields": [],
                "notes": "Groq API key not configured. Cannot auto-fill forms."
            }
        
        # Detect form type
        form_type_prompt = f"""Welk type formulier is relevant voor deze brief? Kies uit: huurtoeslag, zorgtoeslag, duo, gemeente, onbekend.

Brief: {text[:500]}

Antwoord met alleen de naam (huurtoeslag, zorgtoeslag, duo, gemeente, of onbekend):"""

        try:
            type_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": form_type_prompt}],
                temperature=0.1,
                max_tokens=20
            )
            form_type = type_completion.choices[0].message.content.strip().lower()
            
            if form_type not in ['huurtoeslag', 'zorgtoeslag', 'duo', 'gemeente']:
                form_type = 'onbekend'
            
            # Extract fields based on form type
            field_prompt = f"""Extraheer de volgende informatie uit deze overheidsbrief:

Brief: {text}

Extraheer voor formuliertype: {form_type}

Velden om te extraheren (JSON formaat):

{{
  "fields": [
    {{
      "field_name": "naam",
      "field_value": "waarde",
      "confidence": "high/medium/low"
    }}
  ]
}}

Specifieke velden per type:
- huurtoeslag: naam, adres, postcode, woonplaats, huurprijs, BSN, geboortedatum
- zorgtoeslag: naam, adres, postcode, BSN, inkomen, zorgverzekeraar
- duo: naam, BSN, studiefinanciering bedrag, opleiding, instelling
- gemeente: naam, adres, BSN, telefoonnummer, email

Alleen velden invullen die in de brief staan. Geef alleen JSON output."""

            fields_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": field_prompt}],
                temperature=0.2,
                max_tokens=800
            )
            
            result = json.loads(fields_completion.choices[0].message.content)
            
            return {
                "form_type": form_type,
                "fields": result.get("fields", []),
                "notes": "Controleer altijd de ingevulde gegevens. Deze automatische invulling is een hulpmiddel, geen garantie."
            }
            
        except Exception as e:
            print(f"Form extraction error: {e}")
            return {
                "form_type": "onbekend",
                "fields": [],
                "notes": "Automatisch invullen niet mogelijk. Vul het formulier handmatig in."
            }


# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Dutch Bureaucracy Assistant API",
    description="API for analyzing Dutch government letters and auto-filling forms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "https://your-frontend-domain.com",
        "*"  # For development only!
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
ocr_service = OCRService()
llm_service = LLMService()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Dutch Bureaucracy Assistant API",
        "version": "1.0.0",
        "description": "Help Dutch citizens understand government letters",
        "endpoints": {
            "/ocr": "Upload an image to extract text (POST)",
            "/analyze": "Analyze letter text (POST)",
            "/formfill": "Auto-fill forms from letter content (POST)"
        },
        "docs": "/docs",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "tesseract": "available" if ocr_service.tesseract_available else "missing",
            "groq": "configured" if llm_service.client else "missing API key"
        }
    }


@app.post("/ocr", response_model=OCRResponse)
async def extract_text_from_image(file: UploadFile = File(..., description="Image file (JPG or PNG)")):
    """
    Extract text from an uploaded image using OCR.
    
    - Supports JPG and PNG formats
    - Uses Tesseract OCR
    - File is deleted immediately after processing
    """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size (max 5MB)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    max_size = 5 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: 5MB"
        )
    
    try:
        extracted_text = await ocr_service.extract_text_from_image(file)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            return OCRResponse(
                text=extracted_text,
                success=False,
                error="Could not extract sufficient text from the image. Please ensure the image is clear and contains readable text."
            )
        
        return OCRResponse(text=extracted_text, success=True)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_letter(request: AnalyzeRequest):
    """
    Analyze a Dutch government letter.
    
    - Explains the letter in simple Dutch
    - Extracts deadlines with required actions
    - Lists obligations
    - Identifies the government source
    """
    if len(request.text) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Text too long. Maximum 10,000 characters."
        )
    
    try:
        analysis = await llm_service.analyze_letter(request.text)
        
        deadlines = []
        for deadline in analysis.get('deadlines', []):
            deadlines.append({
                "date": deadline.get('date', ''),
                "action": deadline.get('action', 'Neem actie')
            })
        
        return AnalyzeResponse(
            explanation=analysis.get('explanation', 'Kon geen uitleg genereren.'),
            deadlines=deadlines,
            obligations=analysis.get('obligations', []),
            source=analysis.get('source', 'Unknown')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/formfill", response_model=FormFillResponse)
async def auto_fill_form(request: FormFillRequest):
    """
    Extract information from a government letter to auto-fill forms.
    
    - Supports: huurtoeslag, zorgtoeslag, DUO, and gemeente forms
    - Extracts relevant fields automatically
    - Returns confidence levels for each field
    """
    if len(request.text) < 20:
        raise HTTPException(
            status_code=400,
            detail="Text too short. Please provide at least 20 characters of letter content."
        )
    
    if len(request.text) > 15000:
        raise HTTPException(
            status_code=400,
            detail="Text too long. Maximum 15,000 characters."
        )
    
    try:
        extraction_result = await llm_service.extract_form_fields(request.text)
        
        return FormFillResponse(
            form_type=extraction_result.get('form_type', 'onbekend'),
            fields=extraction_result.get('fields', []),
            notes=extraction_result.get('notes', 'Controleer altijd de ingevulde gegevens.')
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Form filling failed: {str(e)}")
        
      @app.post("/process_letter")
async def process_letter(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = None
):
    """
    Combineert OCR, analyse en formfill in één API-call.
    - Als een afbeelding wordt geüpload, gebruikt OCR.
    - Als tekst wordt meegegeven, gebruikt die direct.
    """
    try:
        # 1️⃣ OCR (indien afbeelding)
        if file:
            extracted_text = await ocr_service.extract_text_from_image(file)
        elif text:
            extracted_text = text
        else:
            raise HTTPException(status_code=400, detail="Geen tekst of afbeelding ontvangen.")

        # 2️⃣ Analyse
        analysis = await llm_service.analyze_letter(extracted_text)

        # 3️⃣ Formulier-velden
        form_data = await llm_service.extract_form_fields(extracted_text)

        # 4️⃣ Combineer alles
        return {
            "ocr_text": extracted_text,
            "analysis": analysis,
            "formfill": form_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Verwerking mislukt: {str(e)}")




# ============================================================================
# RUN THE APPLICATION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Dutch Bureaucracy Assistant API")
    print("="*60)
    print("\n📋 Checking dependencies...")
    
    # Check Tesseract
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR: Installed")
    except:
        print("❌ Tesseract OCR: Not found")
        print("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")
    
    # Check Groq API key
    if os.getenv('GROQ_API_KEY'):
        print("✅ Groq API: Configured")
    else:
        print("⚠️  Groq API: No API key found")
        print("   Create .env file with: GROQ_API_KEY=your_key_here")
    
    print("\n🌐 Starting server...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔧 Health Check: http://localhost:8000/health")
    print("\n" + "="*60 + "\n")
    
    uvicorn.run(
        "this_file:app",  # Replace "this_file" with actual filename if different
        host="0.0.0.0",
        port=8000,
        reload=True
    )
