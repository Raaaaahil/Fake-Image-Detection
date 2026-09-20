import { useRef, useState, type ChangeEvent, type DragEvent } from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileImage,
  Loader2,
  RotateCcw,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";

type Prediction = {
  filename: string;
  prediction: string;
  class: "AI" | "REAL";
  confidence: number;
  ai_probability: number;
  real_probability: number;
};

const API_URL = "http://127.0.0.1:8000/predict";

function getConfidenceLevel(confidence: number) {
  if (confidence >= 90) {
    return {
      label: "High confidence",
      className: "bg-emerald-50 text-emerald-700 border-emerald-200",
    };
  }

  if (confidence >= 70) {
    return {
      label: "Moderate confidence",
      className: "bg-amber-50 text-amber-700 border-amber-200",
    };
  }

  return {
    label: "Low confidence",
    className: "bg-slate-50 text-slate-600 border-slate-200",
  };
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  const selectFile = (selectedFile: File) => {
    setError("");
    setResult(null);

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Image must be smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    const url = URL.createObjectURL(selectedFile);
    setPreview(url);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];

    if (selectedFile) {
      selectFile(selectedFile);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);

    const droppedFile = event.dataTransfer.files?.[0];

    if (droppedFile) {
      selectFile(droppedFile);
    }
  };

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(data?.detail || "Unable to analyze image.");
      }

      const data: Prediction = await response.json();
      setResult(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while analyzing the image.",
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview("");
    setResult(null);
    setError("");

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const confidenceInfo = result
    ? getConfidenceLevel(result.confidence)
    : null;

  return (
    <div className="min-h-screen bg-[#f7f8fa] text-slate-900">
      {/* Header */}
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900 text-white">
              <ShieldCheck size={20} />
            </div>

            <div>
              <h1 className="text-base font-semibold tracking-tight">
                TruthLens AI
              </h1>

              <p className="text-xs text-slate-500">
                AI Image Detection
              </p>
            </div>
          </div>

          <div className="hidden items-center gap-2 text-sm text-slate-500 sm:flex">
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
            Model online
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-16">
        {/* Hero */}
        <section className="mx-auto max-w-3xl text-center">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600">
            <ShieldCheck size={14} />
            ResNet-18 powered detection
          </div>

          <h2 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            Is this image{" "}
            <span className="text-slate-500">AI-generated?</span>
          </h2>

          <p className="mx-auto mt-5 max-w-2xl text-base leading-7 text-slate-500">
            Upload an image and TruthLens AI will analyze it to estimate
            whether it was generated by AI or captured naturally.
          </p>
        </section>

        {/* Main workspace */}
        <section className="mx-auto mt-12 max-w-3xl">
          {!result ? (
            <>
              {/* Upload area */}
              {!preview ? (
                <div
                  onDragOver={(event) => {
                    event.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => inputRef.current?.click()}
                  className={`group cursor-pointer rounded-2xl border-2 border-dashed bg-white p-12 text-center transition ${
                    dragging
                      ? "border-slate-900 bg-slate-50"
                      : "border-slate-300 hover:border-slate-500 hover:bg-slate-50"
                  }`}
                >
                  <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-xl bg-slate-100 text-slate-700 transition group-hover:bg-slate-200">
                    <Upload size={24} />
                  </div>

                  <h3 className="mt-5 text-base font-semibold">
                    Drop your image here
                  </h3>

                  <p className="mt-2 text-sm text-slate-500">
                    or click to browse from your device
                  </p>

                  <div className="mt-6 inline-flex rounded-lg bg-slate-900 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-slate-700">
                    Choose image
                  </div>

                  <p className="mt-5 text-xs text-slate-400">
                    JPG, JPEG, PNG or WEBP · Maximum 10 MB
                  </p>

                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                </div>
              ) : (
                /* Image preview */
                <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
                  <div className="relative flex max-h-[520px] items-center justify-center bg-slate-100 p-4">
                    <img
                      src={preview}
                      alt="Selected image"
                      className="max-h-[480px] max-w-full rounded-lg object-contain"
                    />

                    <button
                      type="button"
                      onClick={reset}
                      className="absolute right-5 top-5 flex h-9 w-9 items-center justify-center rounded-full bg-white text-slate-600 shadow-sm transition hover:bg-slate-100"
                      aria-label="Remove image"
                    >
                      <X size={18} />
                    </button>
                  </div>

                  <div className="flex flex-col gap-4 border-t border-slate-200 p-5 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex min-w-0 items-center gap-3">
                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-100">
                        <FileImage size={19} />
                      </div>

                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium">
                          {file?.name}
                        </p>

                        <p className="text-xs text-slate-500">
                          {file
                            ? `${(file.size / 1024 / 1024).toFixed(2)} MB`
                            : ""}
                        </p>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={analyzeImage}
                      disabled={loading}
                      className="flex h-11 items-center justify-center gap-2 rounded-lg bg-slate-900 px-6 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {loading ? (
                        <>
                          <Loader2 size={17} className="animate-spin" />
                          Analyzing...
                        </>
                      ) : (
                        "Analyze image"
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="mt-4 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
                  <AlertCircle size={18} className="mt-0.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}
            </>
          ) : (
            /* Result */
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
              <div className="grid md:grid-cols-2">
                {/* Image */}
                <div className="flex min-h-[360px] items-center justify-center bg-slate-100 p-5">
                  <img
                    src={preview}
                    alt="Analyzed image"
                    className="max-h-[420px] max-w-full rounded-lg object-contain"
                  />
                </div>

                {/* Result details */}
                <div className="p-7 sm:p-9">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Analysis result
                  </p>

                  <div className="mt-6 flex items-start gap-3">
                    <div
                      className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full ${
                        result.class === "AI"
                          ? "bg-amber-100 text-amber-700"
                          : "bg-emerald-100 text-emerald-700"
                      }`}
                    >
                      <CheckCircle2 size={20} />
                    </div>

                    <div className="min-w-0">
                      <h3 className="text-xl font-semibold">
                        {result.prediction}
                      </h3>

                      <div className="mt-1 flex flex-wrap items-center gap-2">
                        <p className="text-sm text-slate-500">
                          {result.confidence.toFixed(2)}% confidence
                        </p>

                        {confidenceInfo && (
                          <span
                            className={`rounded-full border px-2.5 py-1 text-xs font-medium ${confidenceInfo.className}`}
                          >
                            {confidenceInfo.label}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Probability bars */}
                  <div className="mt-8 space-y-5">
                    <ProbabilityBar
                      label="AI generated"
                      value={result.ai_probability}
                      active={result.class === "AI"}
                    />

                    <ProbabilityBar
                      label="Real / natural"
                      value={result.real_probability}
                      active={result.class === "REAL"}
                    />
                  </div>

                  {/* Confidence explanation */}
                  <div className="mt-8 rounded-xl bg-slate-50 p-4">
                    <p className="text-xs leading-5 text-slate-500">
                      Confidence indicates how strongly the model favors its
                      predicted class. Lower confidence means the image is
                      harder for the model to distinguish.
                    </p>
                  </div>

                  {/* General disclaimer */}
                  <div className="mt-3 rounded-xl border border-slate-200 bg-white p-4">
                    <p className="text-xs leading-5 text-slate-500">
                      This result is a model-based estimate. Detection
                      performance may vary across different image sources,
                      editing techniques, and generation methods.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={reset}
                    className="mt-6 flex h-11 w-full items-center justify-center gap-2 rounded-lg border border-slate-300 bg-white text-sm font-medium text-slate-700 transition hover:bg-slate-50"
                  >
                    <RotateCcw size={16} />
                    Analyze another image
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Footer */}
        <section className="mx-auto mt-10 flex max-w-3xl flex-col items-center justify-between gap-3 border-t border-slate-200 pt-6 text-xs text-slate-400 sm:flex-row">
          <span>TruthLens AI · AI Image Detection System</span>
          <span>Model: ResNet-18</span>
        </section>
      </main>
    </div>
  );
}

function ProbabilityBar({
  label,
  value,
  active,
}: {
  label: string;
  value: number;
  active: boolean;
}) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span
          className={`text-sm ${
            active ? "font-medium text-slate-900" : "text-slate-500"
          }`}
        >
          {label}
        </span>

        <span
          className={`text-sm font-semibold ${
            active ? "text-slate-900" : "text-slate-500"
          }`}
        >
          {value.toFixed(2)}%
        </span>
      </div>

      <div className="h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            active ? "bg-slate-900" : "bg-slate-300"
          }`}
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
    </div>
  );
}

export default App;