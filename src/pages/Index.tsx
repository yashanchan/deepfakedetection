import { useState, useEffect } from "react";
import { Shield } from "lucide-react";
import { UploadZone } from "@/components/UploadZone";
import { FilePreview } from "@/components/FilePreview";
import { ProcessingState } from "@/components/ProcessingState";
import { ResultDisplay } from "@/components/ResultDisplay";
import { toast } from "@/hooks/use-toast";

type AppState = "upload" | "preview" | "processing" | "result";

interface AnalysisResult {
  verdict: "REAL" | "FAKE";
  confidence: number;
  processingTime: number;
}

const Index = () => {
  const [state, setState] = useState<AppState>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setState("preview");
    toast({
      title: "File uploaded",
      description: `${file.name} is ready for analysis`,
    });
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setState("upload");
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setState("processing");
    setProgress(0);
    
    const startTime = Date.now();
    const progressInterval = setInterval(() => {
      setProgress((prev) => Math.min(prev + Math.random() * 10, 95));
    }, 300);

    try {
      // Create FormData for file upload
      const formData = new FormData();
      formData.append("file", selectedFile);

      // TODO: Replace with your actual FastAPI backend URL
      const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
      
      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
      }

      const data = await response.json();
      const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);

      clearInterval(progressInterval);
      setProgress(100);

      setResult({
        verdict: data.prediction,
        confidence: Math.round(data.confidence * 100),
        processingTime: parseFloat(processingTime),
      });

      setState("result");

      toast({
        title: "Analysis complete",
        description: `Media classified as ${data.prediction}`,
      });
    } catch (error) {
      clearInterval(progressInterval);
      console.error("Analysis error:", error);
      
      toast({
        title: "Analysis failed",
        description: error instanceof Error ? error.message : "Failed to analyze media. Please try again.",
        variant: "destructive",
      });
      
      setState("preview");
      setProgress(0);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setProgress(0);
    setResult(null);
    setState("upload");
  };

  // Auto-start analysis when file is selected
  useEffect(() => {
    if (state === "preview" && selectedFile) {
      const timer = setTimeout(() => {
        handleAnalyze();
      }, 1500);
      return () => clearTimeout(timer);
    }
  }, [state, selectedFile]);

  return (
    <div className="min-h-screen bg-gradient-mesh">
      {/* Header */}
      <header className="border-b border-border/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-primary/30 rounded-lg blur-lg" />
              <div className="relative bg-gradient-primary p-2 rounded-lg">
                <Shield className="w-6 h-6 text-primary-foreground" />
              </div>
            </div>
            <div>
              <h1 className="text-2xl font-bold">DeepGuard AI</h1>
              <p className="text-sm text-muted-foreground">Advanced Deepfake Detection</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-12">
        <div className="max-w-4xl mx-auto">
          {/* Title Section */}
          <div className="text-center mb-12">
            <h2 className="text-4xl md:text-5xl font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
              Detect AI-Generated Content
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Upload images or videos to analyze authenticity using advanced machine learning
            </p>
          </div>

          {/* Dynamic Content Based on State */}
          <div className="mb-8">
            {state === "upload" && (
              <UploadZone onFileSelect={handleFileSelect} isProcessing={false} />
            )}

            {state === "preview" && selectedFile && (
              <FilePreview file={selectedFile} onRemove={handleRemoveFile} />
            )}

            {state === "processing" && <ProcessingState progress={progress} />}

            {state === "result" && result && (
              <ResultDisplay
                verdict={result.verdict}
                confidence={result.confidence}
                processingTime={result.processingTime}
                onReset={handleReset}
              />
            )}
          </div>

          {/* Info Section */}
          <div className="mt-16 grid md:grid-cols-3 gap-6">
            <div className="p-6 bg-card/50 rounded-xl border border-border">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">AI-Powered Analysis</h3>
              <p className="text-sm text-muted-foreground">
                Uses ResNet-LSTM-Transformer architecture for accurate detection
              </p>
            </div>
            
            <div className="p-6 bg-card/50 rounded-xl border border-border">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">Fast Processing</h3>
              <p className="text-sm text-muted-foreground">
                Get results in seconds with optimized inference pipeline
              </p>
            </div>
            
            <div className="p-6 bg-card/50 rounded-xl border border-border">
              <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold mb-2">Privacy First</h3>
              <p className="text-sm text-muted-foreground">
                Your files are processed securely and never stored permanently
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/50 mt-20">
        <div className="container mx-auto px-6 py-8">
          <p className="text-center text-sm text-muted-foreground">
            DeepGuard AI • Advanced deepfake detection powered by machine learning
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
