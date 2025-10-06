import { useRef, useEffect } from 'react';
import { MessageSquare, Settings, Loader2, Trash2, RefreshCw } from 'lucide-react';
import MessageRenderer from './MessageRenderer';
import ToolsPanel from './ToolsPanel';

interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
}

interface ChatPanelProps {
    sessionId: string | null;
    messages: Message[];
    inputMessage: string;
    isLoading: boolean;
    rightSidebarMode: 'chat' | 'tools';
    apiBase: string;
    onInputChange: (value: string) => void;
    onSendMessage: () => void;
    onClearChat: () => void;
    onNewThread: () => void;
    onSetMode: (mode: 'chat' | 'tools') => void;
    onImageClick: (src: string, alt: string) => void;
}

export default function ChatPanel({
    sessionId,
    messages,
    inputMessage,
    isLoading,
    rightSidebarMode,
    apiBase,
    onInputChange,
    onSendMessage,
    onClearChat,
    onNewThread,
    onSetMode,
    onImageClick
}: ChatPanelProps) {
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    return (
        <div className="w-96 bg-zinc-900/50 border-l border-zinc-800 flex flex-col">
            {/* Mode Toggle */}
            <div className="p-2 border-b border-zinc-800 bg-zinc-900">
                <div className="flex gap-1 bg-zinc-800 rounded-lg p-1">
                    <button
                        onClick={() => onSetMode('chat')}
                        className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${rightSidebarMode === 'chat'
                            ? 'bg-blue-600 text-white'
                            : 'text-zinc-400 hover:text-zinc-300'
                            }`}
                    >
                        <MessageSquare className="h-4 w-4" />
                        Chat
                    </button>
                    <button
                        onClick={() => onSetMode('tools')}
                        className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${rightSidebarMode === 'tools'
                            ? 'bg-blue-600 text-white'
                            : 'text-zinc-400 hover:text-zinc-300'
                            }`}
                    >
                        <Settings className="h-4 w-4" />
                        Tools
                    </button>
                </div>
            </div>

            {/* Chat Mode */}
            {rightSidebarMode === 'chat' && (
                <>
                    <div className="p-4 border-b border-zinc-800">
                        <div className="flex items-center justify-between">
                            <h2 className="text-sm font-semibold text-zinc-400">Chat with AI</h2>
                            <div className="flex gap-2">
                                <button
                                    onClick={onClearChat}
                                    disabled={messages.length === 0}
                                    className="flex items-center gap-1 px-2 py-1 text-xs text-zinc-400 hover:text-zinc-300 hover:bg-zinc-800 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                                    title="Clear chat history"
                                >
                                    <Trash2 className="h-3 w-3" />
                                    Clear
                                </button>
                                <button
                                    onClick={onNewThread}
                                    disabled={!sessionId}
                                    className="flex items-center gap-1 px-2 py-1 text-xs text-zinc-400 hover:text-zinc-300 hover:bg-zinc-800 rounded disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                                    title="Start new conversation thread"
                                >
                                    <RefreshCw className="h-3 w-3" />
                                    New Thread
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="flex-1 overflow-y-auto p-4 space-y-3">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                                {/* Avatar */}
                                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center text-white text-xs font-bold">
                                    {msg.role === 'user' ? '👤' : msg.role === 'system' ? '⚙️' : '🏥'}
                                </div>

                                {/* Message Content */}
                                <div className="flex-1 min-w-0">
                                    <div
                                        className={`inline-block px-4 py-2 rounded-lg text-sm max-w-full ${msg.role === 'user'
                                            ? 'bg-blue-600 text-white'
                                            : msg.role === 'system'
                                                ? 'bg-zinc-800 text-zinc-300'
                                                : 'bg-zinc-800 text-white'
                                            }`}
                                    >
                                        <MessageRenderer
                                            content={msg.content}
                                            apiBase={apiBase}
                                            onImageClick={(src, alt) => onImageClick(src, alt)}
                                        />
                                    </div>
                                    <div className="text-xs text-zinc-600 mt-1">
                                        {msg.timestamp.toLocaleTimeString()}
                                    </div>
                                </div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="p-4 border-t border-zinc-800">
                        <div className="flex gap-2">
                            <input
                                type="text"
                                value={inputMessage}
                                onChange={(e) => onInputChange(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && onSendMessage()}
                                placeholder="Ask about the analysis..."
                                className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm"
                                disabled={isLoading}
                            />
                            <button
                                onClick={onSendMessage}
                                disabled={isLoading}
                                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg disabled:opacity-50"
                            >
                                {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : '→'}
                            </button>
                        </div>
                    </div>
                </>
            )}

            {/* Tools Mode */}
            {rightSidebarMode === 'tools' && <ToolsPanel sessionId={sessionId} />}
        </div>
    );
}

