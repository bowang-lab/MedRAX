import { Settings } from 'lucide-react';

interface PatientInfo {
    name: string;
    age: string;
    gender: string;
    notes: string;
}

interface HeaderProps {
    sessionId: string | null;
    patientInfo: PatientInfo;
    isAnalyzing: boolean;
    showPatientForm: boolean;
    onTogglePatientForm: () => void;
}

export default function Header({
    sessionId,
    patientInfo,
    isAnalyzing,
    showPatientForm,
    onTogglePatientForm
}: HeaderProps) {
    return (
        <div className="bg-zinc-900 border-b border-zinc-800 p-4">
            <div className="max-w-7xl mx-auto">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <div>
                            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-blue-600 bg-clip-text text-transparent">
                                MedRAX Platform
                            </h1>
                            <p className="text-xs text-zinc-500">Medical Image Analysis & Reasoning</p>
                        </div>
                    </div>

                    <div className="flex items-center gap-4">
                        <button
                            onClick={onTogglePatientForm}
                            className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-lg text-sm transition-colors"
                        >
                            {patientInfo.name ? '👤 Patient Info' : '➕ Add Patient Info'}
                        </button>
                        {sessionId && (
                            <div className="text-xs text-zinc-500">
                                Session: {sessionId.slice(0, 8)}
                            </div>
                        )}
                        <div className={`px-3 py-1 rounded-full text-xs font-medium ${isAnalyzing ? 'bg-yellow-900 text-yellow-300' : 'bg-green-900 text-green-300'}`}>
                            {isAnalyzing ? '⏳ Analyzing...' : '✅ Ready'}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

