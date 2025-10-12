'use client';

import { Activity, ChevronLeft, ChevronRight, User, Plus, Calendar, Image as ImageIcon } from 'lucide-react';

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
        <div className={`relative transition-all duration-300 ease-in-out bg-gradient-to-b from-zinc-900 via-zinc-900/95 to-zinc-900/90 border-r border-zinc-800/50 flex flex-col shadow-xl ${collapsed ? 'w-0' : 'w-72'} overflow-hidden`}>
            {/* Header with gradient accent */}
            <div className="relative p-4 border-b border-zinc-800/50 bg-gradient-to-r from-blue-900/20 via-zinc-900/50 to-emerald-900/20">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="p-2 rounded-lg bg-gradient-to-br from-blue-500/10 to-emerald-500/10 border border-blue-500/20">
                            <User className="h-4 w-4 text-blue-400" />
                        </div>
                        <div>
                            <h2 className="text-sm font-bold text-white">Patients</h2>
                            <p className="text-xs text-zinc-500">{sessions.length} recent</p>
                        </div>
                    </div>
                    {onToggleCollapse && (
                        <button
                            onClick={onToggleCollapse}
                            className="p-2 hover:bg-zinc-800/50 rounded-lg text-zinc-400 hover:text-white transition-all duration-200 hover:scale-105"
                            title="Collapse sidebar"
                        >
                            <ChevronLeft className="h-4 w-4" />
                        </button>
                    )}
                </div>
            </div>

            {/* Sessions List with modern styling */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
                {sessions.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full p-6 text-center">
                        <div className="p-4 rounded-full bg-zinc-800/50 mb-3">
                            <User className="h-8 w-8 text-zinc-600" />
                        </div>
                        <p className="text-sm text-zinc-500">No recent patients</p>
                        <p className="text-xs text-zinc-600 mt-1">Start a new case below</p>
                    </div>
                ) : (
                    <div className="p-2 space-y-1">
                        {sessions.map((session) => (
                            <button
                                key={session.sessionId}
                                onClick={() => onSelectSession(session.sessionId)}
                                className={`group w-full p-3 rounded-xl text-left transition-all duration-200 ${session.isActive
                                    ? 'bg-gradient-to-r from-blue-600/20 via-blue-500/10 to-emerald-600/20 border border-blue-500/30 shadow-lg shadow-blue-500/10'
                                    : 'hover:bg-zinc-800/50 border border-transparent hover:border-zinc-700/50'
                                    }`}
                            >
                                {/* Active indicator */}
                                {session.isActive && (
                                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-8 bg-gradient-to-b from-blue-500 to-emerald-500 rounded-r" />
                                )}

                                <div className="flex items-start gap-3">
                                    {/* Avatar */}
                                    <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold ${session.isActive
                                        ? 'bg-gradient-to-br from-blue-500 to-emerald-500 text-white'
                                        : 'bg-zinc-800 text-zinc-400 group-hover:bg-zinc-700'
                                        }`}>
                                        {session.patientName ? session.patientName.charAt(0).toUpperCase() : '?'}
                                    </div>

                                    {/* Info */}
                                    <div className="flex-1 min-w-0">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className={`font-semibold text-sm truncate ${session.isActive ? 'text-white' : 'text-zinc-300'
                                                }`}>
                                                {session.patientName || 'Anonymous Patient'}
                                            </span>
                                            {session.isActive && (
                                                <Activity className="h-3 w-3 text-emerald-400 animate-pulse flex-shrink-0" />
                                            )}
                                        </div>

                                        {/* Metadata */}
                                        <div className="flex items-center gap-3 text-xs text-zinc-500">
                                            {session.patientAge && (
                                                <span className="flex items-center gap-1">
                                                    <Calendar className="h-3 w-3" />
                                                    {session.patientAge}yo
                                                </span>
                                            )}
                                            <span className="flex items-center gap-1">
                                                <ImageIcon className="h-3 w-3" />
                                                {session.imageCount}
                                            </span>
                                        </div>

                                        {/* Timestamp */}
                                        <div className="text-xs text-zinc-600 mt-1">
                                            {new Date(session.timestamp).toLocaleString('en-US', {
                                                month: 'short',
                                                day: 'numeric',
                                                hour: '2-digit',
                                                minute: '2-digit'
                                            })}
                                        </div>
                                    </div>
                                </div>
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* Modern New Patient Button */}
            <div className="p-3 border-t border-zinc-800/50 bg-zinc-900/50 backdrop-blur-sm">
                <button
                    onClick={onNewPatient}
                    className="w-full px-4 py-3 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 rounded-xl text-sm font-semibold transition-all duration-200 hover:scale-105 shadow-lg shadow-blue-500/25 hover:shadow-xl hover:shadow-blue-500/40 flex items-center justify-center gap-2"
                >
                    <Plus className="h-4 w-4" />
                    New Patient Case
                </button>
            </div>
        </div>
    );
}



