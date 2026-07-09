from abc import ABC, abstractmethod
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import requests
import time
from src.utils import get_logger

logger = get_logger("LLMProvider")

# Load environment variables from .env
load_dotenv()

class BaseLLMProvider(ABC):
    """Abstract base class for AI/LLM functionality."""

    @abstractmethod
    def generate_insight(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate an executive business insight based on prompt and analytical context."""
        pass

    @abstractmethod
    def ask_question(self, question: str, analytics_context: Dict[str, Any]) -> str:
        """Answer a chat question from an executive marketing user."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the active provider."""
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Offline mock provider. Guarantees no crashes and works fully offline, returning
    template-based executive summaries when no LLM API key is configured.
    """
    def generate_insight(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        # Generate rich contextual responses if context is provided
        total_rev = "$14.6M"
        overall_roas = "4.25x"
        if context and "revenue_90d" in context:
            rev_val = context["revenue_90d"]
            total_rev = f"${rev_val/1e6:.1f}M" if rev_val >= 1e6 else f"${rev_val:,.0f}"
        if context and "roas_90d" in context:
            overall_roas = f"{context['roas_90d']:.2f}x"

        if "risk" in prompt.lower() or "volatility" in prompt.lower():
            return (
                "[Mock Insight Mode Enabled] Risk Assessment: The portfolio exhibits a Medium risk profile (Score: 42/100). "
                "Google Ads provides high revenue stability, but Meta Ads shows elevated CPC volatility during peak auction hours. "
                "Recommendation: Maintain existing search capture while setting hard CPA caps on Meta prospecting."
            )
        elif "budget" in prompt.lower() or "allocate" in prompt.lower():
            return (
                f"[Mock Insight Mode Enabled] Strategic Budget Recommendations: To maximize total revenue beyond {total_rev}, "
                "shift 15% of underperforming Bing top-of-funnel spend into Google Ads exact match search campaigns. "
                f"This re-allocation is simulated to lift total ROAS from {overall_roas} to 4.65x with 88% confidence."
            )
        elif "causal" in prompt.lower() or "summary" in prompt.lower():
            return (
                f"[Mock Insight Mode Enabled] Causal Inference Summary: Historical revenue growth ({total_rev} projected over 90 days) "
                f"is primarily driven by sustained search capture on Google Ads (53% contribution) and highly efficient remarketing on Meta Ads. "
                f"ROAS efficiency ({overall_roas}) remains highly sensitive to Meta CPM spikes during seasonal high-competition events."
            )
        else:
            return (
                f"[Mock Insight Mode Enabled] Executive AI Analyst Summary: The 90-day multi-channel forecast projects a robust performance "
                f"with total expected revenue reaching {total_rev} and an aggregate ROAS of {overall_roas}. "
                "Performance across Google, Meta, and Bing indicates stable customer acquisition, with Google Search acting as the primary revenue bedrock."
            )

    def ask_question(self, question: str, analytics_context: Dict[str, Any]) -> str:
        q = question.lower()
        if "decrease" in q or "decreasing" in q or "drop" in q or "down" in q:
            return (
                "[Mock Insight Mode Enabled] Causal Analysis: Revenue or ROAS compression is typically linked to aggressive "
                "bidding in Meta Ads during highly competitive auction windows, combined with seasonal dips in search conversion rates. "
                "Our causal inference layer shows a strong negative correlation between Meta CPC inflation and overall daily profitability."
            )
        elif "channel" in q or "growth" in q or "best" in q:
            return (
                "[Mock Insight Mode Enabled] Channel Intelligence: Google Ads is your dominant growth driver, accounting for over 50% "
                "of total baseline revenue with a highly dependable P50 ROAS above 4.5x. Meta Ads acts as your primary volume accelerator "
                "but requires automated budget optimization to mitigate variance."
            )
        elif "meta" in q or "increase" in q or "scale" in q:
            return (
                "[Mock Insight Mode Enabled] Budget Simulation: Increasing Meta Ads spend by 20% is projected to generate an incremental "
                "$340,000 in revenue over the next 60 days. However, marginal efficiency will decrease, slightly reducing Meta ROAS from 3.8x to 3.55x. "
                "This scenario is optimal if your overarching goal is top-line market share expansion."
            )
        else:
            return (
                "[Mock Insight Mode Enabled] Assistant Analyst: Based on your multi-channel marketing data, your current spend efficiency "
                "is exceptionally healthy. To explore specific outcomes, try running a 10,000-run Monte Carlo simulation or use our Optuna budget optimizer "
                "in the navigation tab to discover your exact revenue-maximizing media split."
            )

    def get_provider_name(self) -> str:
        return "MockLLMProvider (100% Offline SaaS Utility)"


class GeminiProvider(BaseLLMProvider):
    """Production provider connecting to Google Gemini API."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"

    def generate_insight(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        full_prompt = f"System: You are an elite Marketing Solutions Architect for ForecastIQ.\nAnalytics: {context}\nPrompt: {prompt}"
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}]
        }
        max_attempts = 3
        backoff = 1
        for attempt in range(max_attempts):
            try:
                res = requests.post(self.endpoint, json=data, timeout=10)
                if res.status_code == 200:
                    return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    logger.warning(f"Gemini API error {res.status_code}: {res.text} (attempt {attempt+1}/{max_attempts})")
            except Exception as e:
                logger.warning(f"Gemini connection exception: {str(e)} (attempt {attempt+1}/{max_attempts})")
            if attempt < max_attempts - 1:
                time.sleep(backoff * (2 ** attempt))
        logger.error("Gemini API failed after %d attempts. Falling back to Mock.", max_attempts)
        return MockLLMProvider().generate_insight(prompt, context)

    def ask_question(self, question: str, analytics_context: Dict[str, Any]) -> str:
        full_prompt = f"System: You are an elite marketing intelligence SaaS chatbot.\nAnalytics: {analytics_context}\nQuestion: {question}"
        data = {
            "contents": [{"parts": [{"text": full_prompt}]}]
        }
        max_attempts = 3
        backoff = 1
        for attempt in range(max_attempts):
            try:
                res = requests.post(self.endpoint, json=data, timeout=10)
                if res.status_code == 200:
                    return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    logger.warning(f"Gemini Chat error {res.status_code} (attempt {attempt+1}/{max_attempts})")
            except Exception as e:
                logger.warning(f"Gemini Chat exception: {str(e)} (attempt {attempt+1}/{max_attempts})")
            if attempt < max_attempts - 1:
                time.sleep(backoff * (2 ** attempt))
        logger.error("Gemini Chat API failed after %d attempts. Falling back to Mock.", max_attempts)
        return MockLLMProvider().ask_question(question, analytics_context)

    def get_provider_name(self) -> str:
        return "Google Gemini (2.5 Flash)"


def get_llm_provider() -> BaseLLMProvider:
    """Factory function that auto-detects API keys from .env or returns MockLLMProvider."""
    if os.getenv("GEMINI_API_KEY"):
        logger.info("Auto-detected GEMINI_API_KEY in environment.")
        return GeminiProvider(os.getenv("GEMINI_API_KEY"))
    else:
        logger.info("No paid API keys detected. Switching to elite MockLLMProvider (100% Offline Mode).")
        return MockLLMProvider()