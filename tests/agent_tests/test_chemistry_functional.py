import pytest

pytestmark = [pytest.mark.unit, pytest.mark.chemistry]


class TestChemistryBasics:
    """Базовые тесты по химии"""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "question,expected_keywords",
        [
            ("Что такое молярная масса?", ["молярная", "моль", "грамм"]),
            (
                "Объясни разницу между ковалентной и ионной связью",
                ["ковалентная", "ионная", "электрон"],
            ),
            ("Какой pH у кислого раствора?", ["pH", "водород", "кислый"]),
        ],
    )
    async def test_chemistry_knowledge(self, invoke_agent, question, expected_keywords):
        """Проверка базовых химических знаний агента"""
        answer = await invoke_agent(question)
        answer_lower = answer.lower()

        # Хотя бы одно ключевое слово должно совпасть
        matches = [kw.lower() in answer_lower for kw in expected_keywords]
        assert any(matches), f"Ответ не содержит ожидаемых терминов. Ответ: {answer}"

    @pytest.mark.asyncio
    async def test_chemistry_calculation(self, invoke_agent):
        """Тест на расчётные задачи (pH, концентрации)"""
        question = "Рассчитай pH 0.01 М раствора HCl"
        answer = await invoke_agent(question)

        # Ожидаем числовой ответ или формулу
        assert any(char.isdigit() for char in answer), "Расчёт должен содержать числа"
        assert "pH" in answer or "водород" in answer.lower(), "Ответ должен относиться к pH"
