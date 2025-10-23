import { Shield, AlertTriangle, CheckCircle, XCircle, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ResultDisplayProps {
  verdict: "REAL" | "FAKE";
  confidence: number;
  processingTime: number;
  onReset: () => void;
}

export const ResultDisplay = ({ verdict, confidence, processingTime, onReset }: ResultDisplayProps) => {
  const isReal = verdict === "REAL";
  const isEdgeCase = confidence >= 40 && confidence <= 60;
  
  const getVerdictColor = () => {
    if (isReal && !isEdgeCase) return "success";
    if (!isReal && !isEdgeCase) return "destructive";
    return "warning";
  };
  
  const color = getVerdictColor();

  return (
    <div className="w-full max-w-2xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-700">
      <div
        className={cn(
          "p-12 rounded-2xl border-2 relative overflow-hidden",
          color === "success" && "bg-success/5 border-success shadow-glow-success",
          color === "destructive" && "bg-destructive/5 border-destructive shadow-glow-danger",
          color === "warning" && "bg-warning/5 border-warning"
        )}
      >
        <div className="absolute top-0 right-0 w-64 h-64 opacity-10">
          <div
            className={cn(
              "w-full h-full rounded-full blur-3xl",
              color === "success" && "bg-success",
              color === "destructive" && "bg-destructive",
              color === "warning" && "bg-warning"
            )}
          />
        </div>
        
        <div className="relative z-10">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <div
                className={cn(
                  "absolute inset-0 rounded-full blur-xl animate-pulse",
                  color === "success" && "bg-success/30",
                  color === "destructive" && "bg-destructive/30",
                  color === "warning" && "bg-warning/30"
                )}
              />
              <div
                className={cn(
                  "relative p-6 rounded-full",
                  color === "success" && "bg-gradient-success",
                  color === "destructive" && "bg-gradient-danger",
                  color === "warning" && "bg-warning"
                )}
              >
                {isReal ? (
                  <CheckCircle className="w-12 h-12 text-white" />
                ) : (
                  <XCircle className="w-12 h-12 text-white" />
                )}
              </div>
            </div>
          </div>
          
          <div className="text-center mb-8">
            <h2
              className={cn(
                "text-5xl font-bold mb-3",
                color === "success" && "text-success",
                color === "destructive" && "text-destructive",
                color === "warning" && "text-warning"
              )}
            >
              {verdict}
            </h2>
            <p className="text-muted-foreground text-lg">
              {isReal
                ? "This media appears to be authentic"
                : "This media shows signs of manipulation"}
            </p>
          </div>
          
          <div className="space-y-6 mb-8">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium">Confidence Level</span>
                <span className="text-sm font-bold">{confidence}%</span>
              </div>
              <div className="h-4 bg-secondary rounded-full overflow-hidden">
                <div
                  className={cn(
                    "h-full transition-all duration-1000",
                    color === "success" && "bg-gradient-success shadow-glow-success",
                    color === "destructive" && "bg-gradient-danger shadow-glow-danger",
                    color === "warning" && "bg-warning"
                  )}
                  style={{ width: `${confidence}%` }}
                />
              </div>
            </div>
            
            {isEdgeCase && (
              <div className="flex items-start gap-3 p-4 bg-warning/10 border border-warning rounded-lg">
                <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
                <div className="text-sm">
                  <p className="font-semibold text-warning mb-1">Edge Case Detected</p>
                  <p className="text-muted-foreground">
                    This result is near the detection threshold. Consider additional verification.
                  </p>
                </div>
              </div>
            )}
            
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Shield className="w-4 h-4" />
              <span>Processed in {processingTime}s using advanced AI detection</span>
            </div>
          </div>
          
          <Button
            onClick={onReset}
            className="w-full"
            size="lg"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Analyze Another File
          </Button>
        </div>
      </div>
    </div>
  );
};
