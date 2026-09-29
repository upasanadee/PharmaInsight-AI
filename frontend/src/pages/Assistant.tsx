import { useState } from "react";
import axios from "axios";
import {
  Bot,
  Send,
  Sparkles,
  AlertTriangle,
  BarChart3,
  Package,
  ShieldAlert,
} from "lucide-react";
import "./Assistant.css";

type AssistantResponse = {
  answer: string;
  source: string;
};

const api = axios.create({
  baseURL: "/api/v1",
});

const suggestedQuestions = [
  {
    icon: ShieldAlert,
    text: "Why is N05C high risk?",
  },
  {
    icon: AlertTriangle,
    text: "Which categories need attention?",
  },
  {
    icon: Package,
    text: "Which categories have intermittent demand?",
  },
  {
    icon: BarChart3,
    text: "Explain the current forecast risks",
  },
];

export default function Assistant() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] =
    useState<AssistantResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const askAssistant = async (questionText?: string) => {
    const query = (questionText ?? question).trim();

    if (!query || loading) {
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await api.post<AssistantResponse>(
        "/assistant",
        {
          question: query,
        }
      );

      setAnswer(response.data);
      setQuestion("");
    } catch (err) {
      console.error(
        "Failed to contact PharmaInsight Assistant:",
        err
      );

      setError(
        "The assistant could not process your question. Please check that the API is running and try again."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (
    event: React.FormEvent<HTMLFormElement>
  ) => {
    event.preventDefault();
    askAssistant();
  };

  return (
    <section className="assistant-page">
      {/* HEADER */}

      <div className="assistant-hero">
        <div className="assistant-icon">
          <Bot size={26} />
        </div>

        <div>
          <span className="eyebrow">
            AI FORECASTING ASSISTANT
          </span>

          <h2>Ask PharmaInsight</h2>

          <p>
            Explore demand forecasts, model performance,
            risk signals and category intelligence using
            the platform's forecasting data.
          </p>
        </div>
      </div>

      {/* SUGGESTED QUESTIONS */}

      <div className="assistant-section">
        <div className="assistant-section-header">
          <Sparkles size={16} />

          <span>Suggested questions</span>
        </div>

        <div className="suggested-grid">
          {suggestedQuestions.map(
            ({ icon: Icon, text }) => (
              <button
                key={text}
                className="suggested-question"
                onClick={() => askAssistant(text)}
                disabled={loading}
              >
                <Icon size={17} />

                <span>{text}</span>
              </button>
            )
          )}
        </div>
      </div>

      {/* QUESTION INPUT */}

      <form
        className="assistant-input-card"
        onSubmit={handleSubmit}
      >
        <div className="input-label">
          <span>Ask a question about your forecasts</span>

          <span className="input-hint">
            Data-grounded assistant
          </span>
        </div>

        <div className="assistant-input-row">
          <input
            type="text"
            value={question}
            onChange={(event) =>
              setQuestion(event.target.value)
            }
            placeholder="e.g. Why is N05C high risk?"
            disabled={loading}
          />

          <button
            type="submit"
            disabled={!question.trim() || loading}
            className="ask-button"
          >
            {loading ? (
              <>
                <span className="assistant-spinner" />
                Thinking
              </>
            ) : (
              <>
                <Send size={16} />
                Ask
              </>
            )}
          </button>
        </div>
      </form>

      {/* ERROR */}

      {error && (
        <div className="assistant-error">
          <AlertTriangle size={18} />

          <span>{error}</span>
        </div>
      )}

      {/* RESPONSE */}

      {answer && !error && (
        <div className="assistant-response">
          <div className="response-header">
            <div className="response-icon">
              <Bot size={19} />
            </div>

            <div>
              <strong>PharmaInsight Assistant</strong>

              <span>
                Source: {answer.source}
              </span>
            </div>
          </div>

          <div className="response-body">
            {answer.answer
              .split("\n")
              .map((line, index) => (
                <p key={index}>
                  {line || "\u00A0"}
                </p>
              ))}
          </div>
        </div>
      )}

      {/* EMPTY STATE */}

      {!answer && !error && !loading && (
        <div className="assistant-empty">
          <Bot size={30} />

          <h3>Ask about your pharmaceutical demand data</h3>

          <p>
            The assistant can explain forecast risks,
            demand regimes, category performance and
            other intelligence generated by PharmaInsight.
          </p>
        </div>
      )}
    </section>
  );
}

