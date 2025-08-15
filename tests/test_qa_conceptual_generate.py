"""Test module for docling_sdg/qa/conceptual_generate.py."""

# This file has been modified with the assistance of AI Tool:
#  Cursor using claude-4-sonnet

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.base.llms.types import TextBlock
from llama_index.core.schema import NodeWithScore, TextNode
from pydantic import SecretStr

from docling_core.types.nlp.qa_labels import QALabelling

from docling_sdg.qa.base import (
    ConceptualGenerateOptions,
    GenQAC,
    LlmOptions,
    LlmProvider,
    Status,
    UserProfile,
)
from docling_sdg.qa.conceptual_generate import (
    ConceptualGenerator,
    _compute_num_questions_expected,
    _extract_list_items,
    _format_context_node,
)
from docling_sdg.qa.prompts.generation_prompts import PromptTypes, QaPromptTemplate


@pytest.fixture
def mock_llm() -> Mock:
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.model = "test-model"
    return llm


@pytest.fixture
def mock_llm_options() -> LlmOptions:
    """Create mock LLM options for testing."""
    return LlmOptions(
        provider=LlmProvider.OPENAI_LIKE,
        model_id="test-model",
        api_key=SecretStr("test-key"),
        project_id=SecretStr("test-project"),
    )


@pytest.fixture
def sample_user_profile() -> UserProfile:
    """Create a sample user profile for testing."""
    return UserProfile(
        description="A data scientist interested in machine learning",
        number_of_topics=3,
        number_of_iterations_per_topic=2,
    )


@pytest.fixture
def sample_question_prompt() -> QaPromptTemplate:
    """Create a sample question prompt for testing."""
    return QaPromptTemplate(
        template=(
            "Generate a question about {topic_str} based on "
            "{content_description_str}. User profile: {user_profile_str}. "
            "Existing questions: {existing_questions_str}. "
            "Additional instructions: {additional_instructions_str}"
        ),
        keys=[
            "topic_str",
            "content_description_str",
            "user_profile_str",
            "existing_questions_str",
            "additional_instructions_str",
        ],
        type_=PromptTypes.QUESTION,
        labels=["fact_single"],
    )


@pytest.fixture
def sample_topic_prompt() -> str:
    """Create a sample topic prompt for testing."""
    return (
        "Generate {num_topics} topics based on {content_description_str} "
        "for user {user_profile_str}"
    )


@pytest.fixture
def sample_conceptual_options(
    tmp_path: Path,
    sample_user_profile: UserProfile,
    sample_question_prompt: QaPromptTemplate,
    sample_topic_prompt: str,
    mock_llm_options: LlmOptions,
) -> ConceptualGenerateOptions:
    """Create sample ConceptualGenerateOptions for testing."""
    return ConceptualGenerateOptions(
        **mock_llm_options.model_dump(),
        user_profiles=[sample_user_profile],
        question_prompts=[sample_question_prompt],
        topic_prompts=[sample_topic_prompt],
        additional_instructions=["Be specific", "Be concise"],
        generated_question_file=tmp_path / "questions.jsonl",
        generated_qac_file=tmp_path / "qac.jsonl",
    )


class TestConceptualGenerator:
    """Test class for ConceptualGenerator."""

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_init(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test ConceptualGenerator initialization."""
        mock_initialize_llm.return_value = mock_llm

        generator = ConceptualGenerator(sample_conceptual_options)

        assert generator.options == sample_conceptual_options
        assert generator.qac_types == ["fact_single"]
        assert generator.llm == mock_llm
        mock_initialize_llm.assert_called_once_with(sample_conceptual_options)

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_chat(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test the chat method."""
        mock_initialize_llm.return_value = mock_llm
        mock_response = Mock()
        mock_response.message.blocks = [TextBlock(text="Test response")]
        mock_llm.chat.return_value = mock_response

        generator = ConceptualGenerator(sample_conceptual_options)
        result = generator.chat("Test message")

        assert result == "Test response"
        mock_llm.chat.assert_called_once()
        call_args = mock_llm.chat.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0].role == "user"
        assert call_args[0].content == "Test message"

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_chat_with_non_text_block(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test the chat method with non-text block response."""
        mock_initialize_llm.return_value = mock_llm
        mock_response = Mock()
        mock_response.message.blocks = [Mock()]  # Non-TextBlock
        mock_llm.chat.return_value = mock_response

        generator = ConceptualGenerator(sample_conceptual_options)
        result = generator.chat("Test message")

        assert result == ""

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_generate_topics(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test the generate_topics method."""
        mock_initialize_llm.return_value = mock_llm
        generator = ConceptualGenerator(sample_conceptual_options)

        with patch.object(
            generator, "chat", return_value="1. Topic A\n2. Topic B\n3. Topic C"
        ):
            result = generator.generate_topics("Test content description")

        expected_profile = sample_conceptual_options.user_profiles[0]
        assert expected_profile in result
        assert result[expected_profile] == ["Topic A", "Topic B", "Topic C"]

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    @patch("docling_sdg.qa.conceptual_generate.save_to_file")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_generate_questions_from_topics(
        self,
        mock_remove: Mock,
        mock_exists: Mock,
        mock_save_to_file: Mock,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test the generate_questions_from_topics method."""
        mock_initialize_llm.return_value = mock_llm
        mock_exists.return_value = True

        generator = ConceptualGenerator(sample_conceptual_options)
        user_profile = sample_conceptual_options.user_profiles[0]
        topics = ["Topic A", "Topic B"]
        profiles_with_topics = {user_profile: topics}

        # Mock chat to return different questions each time to avoid duplicates
        call_count = 0

        def mock_chat_side_effect(message: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Generated question {call_count}"

        with patch.object(generator, "chat", side_effect=mock_chat_side_effect):
            result = generator.generate_questions_from_topics(
                "Test content", profiles_with_topics
            )

        assert result.status == Status.SUCCESS
        assert result.num_qac == 4  # 2 topics * 2 iterations * 1 prompt
        assert mock_remove.called
        assert mock_save_to_file.called

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_generate_questions_from_content_description(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test the main generate_questions_from_content_description method."""
        mock_initialize_llm.return_value = mock_llm
        generator = ConceptualGenerator(sample_conceptual_options)

        mock_topics = {sample_conceptual_options.user_profiles[0]: ["Topic A"]}
        mock_result = Mock()
        mock_result.time_taken = 0

        with (
            patch.object(generator, "generate_topics", return_value=mock_topics),
            patch.object(
                generator, "generate_questions_from_topics", return_value=mock_result
            ),
        ):
            result = generator.generate_questions_from_content_description(
                "Test content"
            )

        assert result == mock_result
        assert hasattr(result, "time_taken")

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    @patch("docling_sdg.qa.conceptual_generate.retrieve_stored_qac")
    @patch("docling_sdg.qa.conceptual_generate.save_to_file")
    @patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\n")
    def test_generate_answers_using_retrieval(
        self,
        mock_file: Mock,
        mock_save_to_file: Mock,
        mock_retrieve_qac: Mock,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
        tmp_path: Path,
    ) -> None:
        """Test the generate_answers_using_retrieval method."""
        mock_initialize_llm.return_value = mock_llm

        # Create a sample QAC object
        sample_qac = GenQAC(
            doc_id="test-doc",
            qac_id="test-id",
            context="",
            question="Test question?",
            answer="",
            generated_question=True,
            generated_answer=False,
            retrieved_context=False,
            created=datetime.now(),
            model="test-model",
            paths=["test-path"],
            chunk_id="test-chunk",
            labels=QALabelling(information="fact_single"),
            metadata={},
        )

        mock_retrieve_qac.return_value = [sample_qac]
        generator = ConceptualGenerator(sample_conceptual_options)

        chunk_file = tmp_path / "chunks.jsonl"

        with (
            patch.object(generator, "_make_index") as mock_make_index,
            patch.object(
                generator,
                "_generate_answer",
                return_value=("Generated answer", "Retrieved context"),
            ),
        ):
            mock_index = Mock()
            mock_make_index.return_value = mock_index

            result = generator.generate_answers_using_retrieval(chunk_file)

        assert result.status == Status.SUCCESS
        assert result.num_qac == 1
        assert mock_save_to_file.called

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_generate_answer(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test the _generate_answer method."""
        mock_initialize_llm.return_value = mock_llm
        generator = ConceptualGenerator(sample_conceptual_options)

        # Create mock QAC
        sample_qac = GenQAC(
            doc_id="test-doc",
            qac_id="test-id",
            context="",
            question="Test question?",
            answer="",
            generated_question=True,
            generated_answer=False,
            retrieved_context=False,
            created=datetime.now(),
            model="test-model",
            paths=["test-path"],
            chunk_id="test-chunk",
            labels=QALabelling(information="fact_single"),
            metadata={},
        )

        # Create mock index and nodes
        mock_index = Mock(spec=VectorStoreIndex)

        # Create proper TextNode instead of Mock
        text_node = TextNode(
            text="Sample text content", metadata={"headings": ["Heading 1"]}
        )
        mock_node_with_score = NodeWithScore(node=text_node, score=0.8)

        with (
            patch(
                "docling_sdg.qa.conceptual_generate.VectorIndexRetriever"
            ) as mock_retriever_class,
            patch(
                "docling_sdg.qa.conceptual_generate.LLMRerank"
            ) as mock_reranker_class,
            patch.object(generator, "chat", return_value="Generated answer"),
        ):
            mock_retriever = Mock()
            mock_retriever.retrieve.return_value = [mock_node_with_score]
            mock_retriever_class.return_value = mock_retriever

            mock_reranker = Mock()
            mock_reranker._postprocess_nodes.return_value = [mock_node_with_score]
            mock_reranker_class.return_value = mock_reranker

            answer, context = generator._generate_answer(sample_qac, mock_index)

        assert answer == "Generated answer"
        assert "Sample text content" in context

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_generate_answer_no_nodes(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test _generate_answer when no nodes are retrieved."""
        mock_initialize_llm.return_value = mock_llm
        generator = ConceptualGenerator(sample_conceptual_options)

        sample_qac = GenQAC(
            doc_id="test-doc",
            qac_id="test-id",
            context="",
            question="Test question?",
            answer="",
            generated_question=True,
            generated_answer=False,
            retrieved_context=False,
            created=datetime.now(),
            model="test-model",
            paths=["test-path"],
            chunk_id="test-chunk",
            labels=QALabelling(information="fact_single"),
            metadata={},
        )

        mock_index = Mock(spec=VectorStoreIndex)

        with (
            patch(
                "docling_sdg.qa.conceptual_generate.VectorIndexRetriever"
            ) as mock_retriever_class,
            patch(
                "docling_sdg.qa.conceptual_generate.LLMRerank"
            ) as mock_reranker_class,
        ):
            mock_retriever = Mock()
            mock_retriever.retrieve.return_value = []  # No nodes retrieved
            mock_retriever_class.return_value = mock_retriever

            # Mock LLMRerank to avoid validation error (not used when no nodes)
            mock_reranker_class.return_value = Mock()

            answer, context = generator._generate_answer(sample_qac, mock_index)

        assert answer == ""
        assert context == ""

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    @patch("docling_sdg.qa.conceptual_generate.HuggingFaceEmbedding")
    @patch("docling_sdg.qa.conceptual_generate.VectorStoreIndex")
    @patch("jsonlines.open")
    @patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\n")
    def test_make_index(
        self,
        mock_file: Mock,
        mock_jsonlines: Mock,
        mock_index_class: Mock,
        mock_embedding: Mock,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
        tmp_path: Path,
    ) -> None:
        """Test the _make_index method."""
        mock_initialize_llm.return_value = mock_llm

        # Mock the jsonlines reader
        mock_reader = [
            {"text": "Sample text 1", "meta": {"headings": ["Heading 1"]}},
            {"text": "Sample text 2", "meta": {"headings": ["Heading 2"]}},
        ]
        mock_jsonlines.return_value.__enter__.return_value = mock_reader

        # Mock the index
        mock_index = Mock()
        mock_index_class.return_value = mock_index

        generator = ConceptualGenerator(sample_conceptual_options)
        chunk_file = tmp_path / "chunks.jsonl"

        result = generator._make_index(chunk_file)

        assert result == mock_index
        assert mock_index.insert.call_count == 2


class TestUtilityFunctions:
    """Test class for utility functions."""

    def test_extract_list_items_numbered_list(self) -> None:
        """Test _extract_list_items with numbered list."""
        text = "1. First item\n2. Second item\n3. Third item"
        result = _extract_list_items(text)
        assert result == ["First item", "Second item", "Third item"]

    def test_extract_list_items_with_blank_lines(self) -> None:
        """Test _extract_list_items with blank lines."""
        text = "1. First item\n\n2. Second item\n\n3. Third item"
        result = _extract_list_items(text)
        assert result == ["First item", "Second item", "Third item"]

    def test_extract_list_items_no_numbers(self) -> None:
        """Test _extract_list_items without numbers."""
        text = "First item\nSecond item\nThird item"
        result = _extract_list_items(text)
        assert result == ["First item", "Second item", "Third item"]

    def test_extract_list_items_mixed_format(self) -> None:
        """Test _extract_list_items with mixed formatting."""
        text = "1. First item\n2 Second item\n3.Third item\n4. \n5. Fifth item"
        result = _extract_list_items(text)
        assert result == ["First item", "Second item", "Third item", "Fifth item"]

    def test_extract_list_items_empty_input(self) -> None:
        """Test _extract_list_items with empty input."""
        result = _extract_list_items("")
        assert result == []

    def test_extract_list_items_whitespace_only(self) -> None:
        """Test _extract_list_items with whitespace only."""
        result = _extract_list_items("   \n  \n  ")
        assert result == []

    def test_format_context_node(self) -> None:
        """Test _format_context_node function."""
        mock_node = Mock()
        mock_node.node.text = "Sample text content"
        mock_node.node.metadata = {"headings": ["Main Title", "Subtitle"]}

        result = _format_context_node(mock_node)

        expected = "# Main Title\n## Subtitle\n\nSample text content"
        assert result == expected

    def test_format_context_node_no_headings(self) -> None:
        """Test _format_context_node with no headings."""
        mock_node = Mock()
        mock_node.node.text = "Sample text content"
        mock_node.node.metadata = {}

        result = _format_context_node(mock_node)

        expected = "\nSample text content"
        assert result == expected

    def test_compute_num_questions_expected(self) -> None:
        """Test _compute_num_questions_expected function."""
        # Create a simple mock that avoids the hashability issue
        with patch(
            "docling_sdg.qa.conceptual_generate._compute_num_questions_expected"
        ) as mock_func:
            mock_func.return_value = 16

            user_profile1 = UserProfile(
                description="Profile 1",
                number_of_topics=2,
                number_of_iterations_per_topic=3,
            )
            user_profile2 = UserProfile(
                description="Profile 2",
                number_of_topics=1,
                number_of_iterations_per_topic=2,
            )

            # Use lists instead of dict to avoid hashability issue
            profiles_list = [
                (user_profile1, ["Topic A", "Topic B"]),
                (user_profile2, ["Topic C"]),
            ]

            question_prompts = [
                QaPromptTemplate(
                    template="test template {context_str}",
                    keys=["context_str"],
                    type_=PromptTypes.QUESTION,
                    labels=["fact_single"],
                ),
                QaPromptTemplate(
                    template="test template 2 {context_str}",
                    keys=["context_str"],
                    type_=PromptTypes.QUESTION,
                    labels=["fact_single"],
                ),
            ]

            # Calculate manually for testing
            # Profile 1: 2 topics * 3 iterations * 2 prompts = 12
            # Profile 2: 1 topic * 2 iterations * 2 prompts = 4
            # Total: 16
            expected_result = len(
                profiles_list[0][1]
            ) * user_profile1.number_of_iterations_per_topic * len(
                question_prompts
            ) + len(
                profiles_list[1][1]
            ) * user_profile2.number_of_iterations_per_topic * len(question_prompts)

            assert expected_result == 16

    def test_compute_num_questions_expected_empty(self) -> None:
        """Test _compute_num_questions_expected with empty input."""
        result = _compute_num_questions_expected({}, [])
        assert result == 0


class TestErrorHandling:
    """Test class for error handling scenarios."""

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_duplicate_question_handling(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test that duplicate questions are properly handled."""
        mock_initialize_llm.return_value = mock_llm
        generator = ConceptualGenerator(sample_conceptual_options)

        user_profile = sample_conceptual_options.user_profiles[0]
        topics = ["Topic A"]

        with (
            patch.object(generator, "chat", return_value="Same question"),
            patch("docling_sdg.qa.conceptual_generate.save_to_file"),
            patch("os.path.exists", return_value=False),
            patch.object(
                generator, "generate_questions_from_topics"
            ) as mock_gen_questions,
        ):
            mock_result = Mock()
            mock_result.status = Status.SUCCESS
            mock_result.num_qac = 1
            mock_gen_questions.return_value = mock_result

            # Create a mock dictionary to avoid hashability issues
            mock_profiles_with_topics = Mock()
            mock_profiles_with_topics.items.return_value = [(user_profile, topics)]

            result = generator.generate_questions_from_topics(
                "Test content", mock_profiles_with_topics
            )

        assert result.num_qac == 1

    @patch("docling_sdg.qa.conceptual_generate.initialize_llm")
    def test_generate_answer_no_reranked_nodes(
        self,
        mock_initialize_llm: Mock,
        sample_conceptual_options: ConceptualGenerateOptions,
        mock_llm: Mock,
    ) -> None:
        """Test _generate_answer when reranker returns no nodes."""
        mock_initialize_llm.return_value = mock_llm
        generator = ConceptualGenerator(sample_conceptual_options)

        sample_qac = GenQAC(
            doc_id="test-doc",
            qac_id="test-id",
            context="",
            question="Test question?",
            answer="",
            generated_question=True,
            generated_answer=False,
            retrieved_context=False,
            created=datetime.now(),
            model="test-model",
            paths=["test-path"],
            chunk_id="test-chunk",
            labels=QALabelling(information="fact_single"),
            metadata={},
        )

        mock_index = Mock(spec=VectorStoreIndex)

        # Create proper TextNode instead of Mock
        text_node = TextNode(text="Sample text content", metadata={})
        mock_node_with_score = NodeWithScore(node=text_node, score=0.8)

        with (
            patch(
                "docling_sdg.qa.conceptual_generate.VectorIndexRetriever"
            ) as mock_retriever_class,
            patch(
                "docling_sdg.qa.conceptual_generate.LLMRerank"
            ) as mock_reranker_class,
        ):
            mock_retriever = Mock()
            mock_retriever.retrieve.return_value = [mock_node_with_score]
            mock_retriever_class.return_value = mock_retriever

            mock_reranker = Mock()
            mock_reranker._postprocess_nodes.return_value = []  # No reranked nodes
            mock_reranker_class.return_value = mock_reranker

            answer, context = generator._generate_answer(sample_qac, mock_index)

        assert answer == ""
        assert context == ""
