'use client';

import { Activity, ChevronLeft, ChevronRight } from 'lucide-react';

interface PatientSession {
    sessionId: string;
    patientName: string;
    patientAge: string;
    timestamp: Date;
    imageCount: number;
    isActive: boolean;
}

interface PatientSidebarProps {
    sessions: PatientSession[];
    onSelectSession: (sessionId: string) => void;
    onNewPatient: () => void;
    collapsed?: boolean;
    onToggleCollapse?: () => void;
}

export default function PatientSidebar({ sessions, onSelectSession, onNewPatient, collapsed = false, onToggleCollapse }: PatientSidebarProps) {
    return (
        <div className={`relative transition-all duration-300 ease-in-out bg-zinc-900 border-r border-zinc-800 flex flex-col ${collapsed ? 'w-0' : 'w-64'} overflow-hidden`}>
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-zinc-400">Recent Patients</h2>
                {onToggleCollapse && (
                    <button
                        onClick={onToggleCollapse}
                        className="p-1 hover:bg-zinc-800 rounded text-zinc-400 hover:text-zinc-300 transition-colors"
                        title="Collapse sidebar"
                    >
                        <ChevronLeft className="h-4 w-4" />
                    </button>
                )}
            </div>

            <div className="flex-1 overflow-y-auto">
                {sessions.length === 0 ? (
                    <div className="p-4 text-sm text-zinc-500 text-center">
                        No previous patients
                    </div>
                ) : (
                    sessions.map((session) => (
                        <button
                            key={session.sessionId}
                            onClick={() => onSelectSession(session.sessionId)}
                            className={`w-full p-4 text-left border-b border-zinc-800 hover:bg-zinc-800/50 transition-colors ${session.isActive ? 'bg-zinc-800 border-l-2 border-l-blue-500' : ''
                                }`}
                        >
                            <div className="flex items-center gap-2 mb-1">
                                {session.isActive && <Activity className="h-3 w-3 text-blue-500" />}
                                <span className="font-medium text-sm">
                                    {session.patientName || 'Anonymous'}
                                    {session.patientAge && `, ${session.patientAge}yo`}
                                </span>
                            </div>
                            <div className="text-xs text-zinc-500">
                                {session.imageCount} image{session.imageCount !== 1 ? 's' : ''}
                                {' • '}
                                {new Date(session.timestamp).toLocaleTimeString()}
                            </div>
                        </button>
                    ))
                )}
            </div>

            <div className="p-4 border-t border-zinc-800">
                <button
                    onClick={onNewPatient}
                    className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm font-medium transition-colors"
                >
                    🏥 New Patient
                </button>
            </div>
        </div>
    );
}



