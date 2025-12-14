from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPEN_AI_API_KEY")
)

# Knowledge base per date type
DATE_INFO = {
    "Ajwa": {
        "name_ar": "عجوة",
        "origin": "مشهور في المدينة المنورة",
        "color": "غامق يميل للسواد",
        "texture": "طري وناعم",
        "taste": "حلاوة متوازنة بدون ما تكون مزعجة",
        "reputation": "يعتبر تمر فخم ومحبوب عند كثير من الناس"
    },
    "Medjool": {
        "name_ar": "مجدول",
        "origin": "منتشر عالميًا ويعتبر تمر فاخر",
        "color": "بني متوسط لداكن",
        "texture": "كبير الحجم ولحم التمرة ثقيل",
        "taste": "حالي جدًا وطعمه يشبه الكراميل",
        "reputation": "يستخدم غالبًا في الضيافة والطلبات الفخمة"
    },
    "Sokari": {
        "name_ar": "سكري",
        "origin": "مشهور خاصة عند أهل القصيم",
        "color": "ذهبي إلى بني فاتح",
        "texture": "مقرمش إذا كان خراف أو طري إذا كان رطب",
        "taste": "حالي بقوة، اسمه جاي من السكر",
        "reputation": "أساسي مع القهوة عند كثير من أهل السعودية"
    },
    "Sugaey": {
        "name_ar": "صقعي",
        "origin": "مشهور في الرياض والقصيم",
        "color": "أطرافه فاتحة ووسطه أغمق شوي",
        "texture": "حبة متوسطة إلى كبيرة وقوامها متوسط",
        "taste": "حلاوته متوسطة، لا هو خفيف ولا ثقيل",
        "reputation": "يجي كثير مع خلطات التمور والضيافة"
    }
}


def build_date_description_prompt(date_type: str) -> str:
    """Build a structured Saudi-dialect prompt for the given date type."""

    info = DATE_INFO.get(date_type, None)

    if info is None:
        type_ar = date_type
        extra_hint = ""
    else:
        type_ar = info["name_ar"]
        extra_hint = f"""
        معلومات عن هذا النوع (لا تكررها حرفيًا، استخدمها كخلفية فقط):
        - الأصل: {info['origin']}
        - اللون: {info['color']}
        - القوام: {info['texture']}
        - الطعم: {info['taste']}
        - سمعته: {info['reputation']}
        """

    prompt = f"""
    تخيّل نفسك شاب سعودي يوصف تمر {type_ar} لصاحبه.

    {extra_hint}

    اكتب بالضبط 5 جمل، كل جملة في سطر جديد.
    كل جملة أقل من 10 كلمات.

    الأسلوب:
    - لهجة سعودية بسيطة، كلام شباب، بدون رسمية.
    - لا تستعمل فصحى معقدة ولا كلمات غريبة.

    هيكلة الجمل:
    - الجملة الأولى: تعريف سريع بالتمر وانطباع عام.
    - الجملة الثانية: وصف مختصر للون والشكل.
    - الجملة الثالثة: الطعم والقوام، خصوصًا أول لقمة.
    - الجملة الرابعة: كيف يأكلونه الناس عادة في السعودية.
    - الجملة الخامسة: تعليق شبابي طريف أو رأي شخصي قصير.

    ممنوع:
    - لا تكتب تعداد نقطي أو أرقام.
    - لا تكتب أسئلة.
    - لا تشرح التعليمات أو تذكر أنك نموذج لغة.
    - لا تضيف أي جملة سادسة أو زيادة.

    الآن اكتب الخمس جمل فقط، كل جملة في سطر مستقل.
    """

    return prompt.strip()


def generate_text(
    prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.7,
    max_tokens: int = 200,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "أنت مساعد ذكي مختص بوصف التمور."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,  # ✅ what the API is asking for
        )

        text = response.choices[0].message.content
        print("[LLM] Generated text:\n", text)
        return text

    except Exception as e:
        print("[LLM] Error during generation:", e)
        return "تعذر إنشاء الوصف حالياً. حاول مرة ثانية لاحقاً."
