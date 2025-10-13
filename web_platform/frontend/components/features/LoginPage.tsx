'use client';

import { useState } from 'react';
import { User, Lock, LogIn, UserPlus } from 'lucide-react';
import axios from 'axios';

interface LoginPageProps {
    onLoginSuccess: (token: string, user: any) => void;
    apiBase: string;
}

export default function LoginPage({ onLoginSuccess, apiBase }: LoginPageProps) {
    const [isRegister, setIsRegister] = useState(false);
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [displayName, setDisplayName] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setLoading(true);

        try {
            if (isRegister) {
                // Register
                await axios.post(`${apiBase}/api/auth/register`, {
                    username,
                    password,
                    display_name: displayName || username
                });

                // Auto-login after registration
                const loginResponse = await axios.post(`${apiBase}/api/auth/login`, {
                    username,
                    password
                });

                const { token, user } = loginResponse.data;
                localStorage.setItem('medrax_token', token);
                localStorage.setItem('medrax_user', JSON.stringify(user));
                onLoginSuccess(token, user);
            } else {
                // Login
                const response = await axios.post(`${apiBase}/api/auth/login`, {
                    username,
                    password
                });

                const { token, user } = response.data;
                localStorage.setItem('medrax_token', token);
                localStorage.setItem('medrax_user', JSON.stringify(user));
                onLoginSuccess(token, user);
            }
        } catch (err: any) {
            setError(err.response?.data?.detail || 'Authentication failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-zinc-950 via-zinc-900 to-zinc-950">
            <div className="w-full max-w-md p-8">
                {/* Logo/Header */}
                <div className="text-center mb-8">
                    <div className="inline-block p-4 bg-gradient-to-br from-blue-600 to-emerald-600 rounded-2xl mb-4">
                        <User className="h-12 w-12 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold text-white mb-2">MedRAX Platform</h1>
                    <p className="text-zinc-400">Medical Imaging AI Assistant</p>
                </div>

                {/* Auth Form */}
                <div className="bg-zinc-900/50 backdrop-blur-sm border border-zinc-800 rounded-2xl p-8 shadow-2xl">
                    <div className="flex gap-2 mb-6">
                        <button
                            onClick={() => setIsRegister(false)}
                            className={`flex-1 py-2 px-4 rounded-xl font-medium transition-all ${!isRegister
                                ? 'bg-gradient-to-r from-blue-600 to-emerald-600 text-white shadow-lg'
                                : 'bg-zinc-800/50 text-zinc-400 hover:text-zinc-300'
                                }`}
                        >
                            Login
                        </button>
                        <button
                            onClick={() => setIsRegister(true)}
                            className={`flex-1 py-2 px-4 rounded-xl font-medium transition-all ${isRegister
                                ? 'bg-gradient-to-r from-blue-600 to-emerald-600 text-white shadow-lg'
                                : 'bg-zinc-800/50 text-zinc-400 hover:text-zinc-300'
                                }`}
                        >
                            Register
                        </button>
                    </div>

                    <form onSubmit={handleSubmit} className="space-y-4">
                        {/* Username */}
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">
                                Username
                            </label>
                            <div className="relative">
                                <User className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-zinc-500" />
                                <input
                                    type="text"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    className="w-full pl-10 pr-4 py-3 bg-zinc-800/50 border border-zinc-700 rounded-xl text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent"
                                    placeholder="Enter username"
                                    required
                                    minLength={3}
                                />
                            </div>
                        </div>

                        {/* Display Name (Register only) */}
                        {isRegister && (
                            <div>
                                <label className="block text-sm font-medium text-zinc-300 mb-2">
                                    Display Name (Optional)
                                </label>
                                <input
                                    type="text"
                                    value={displayName}
                                    onChange={(e) => setDisplayName(e.target.value)}
                                    className="w-full px-4 py-3 bg-zinc-800/50 border border-zinc-700 rounded-xl text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent"
                                    placeholder="Your name"
                                />
                            </div>
                        )}

                        {/* Password */}
                        <div>
                            <label className="block text-sm font-medium text-zinc-300 mb-2">
                                Password
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-zinc-500" />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    className="w-full pl-10 pr-4 py-3 bg-zinc-800/50 border border-zinc-700 rounded-xl text-white placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-transparent"
                                    placeholder="Enter password"
                                    required
                                />
                            </div>
                            {isRegister && (
                                <p className="mt-1 text-xs text-zinc-500">
                                    Any password is fine - no strength requirements
                                </p>
                            )}
                        </div>

                        {/* Error Message */}
                        {error && (
                            <div className="p-3 bg-red-500/10 border border-red-500/50 rounded-xl text-red-400 text-sm">
                                {error}
                            </div>
                        )}

                        {/* Submit Button */}
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 bg-gradient-to-r from-blue-600 to-emerald-600 hover:from-blue-500 hover:to-emerald-500 text-white font-semibold rounded-xl transition-all duration-200 hover:scale-105 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <div className="h-5 w-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    {isRegister ? 'Creating Account...' : 'Logging In...'}
                                </>
                            ) : (
                                <>
                                    {isRegister ? <UserPlus className="h-5 w-5" /> : <LogIn className="h-5 w-5" />}
                                    {isRegister ? 'Create Account' : 'Login'}
                                </>
                            )}
                        </button>
                    </form>

                    {/* Info */}
                    <div className="mt-6 text-center text-xs text-zinc-500">
                        {isRegister ? (
                            <p>Already have an account? Click Login above</p>
                        ) : (
                            <p>Don't have an account? Click Register above</p>
                        )}
                    </div>
                </div>

                {/* Footer */}
                <div className="mt-6 text-center text-xs text-zinc-600">
                    <p>Simple authentication for multi-user access</p>
                    <p className="mt-1">Each user has isolated data and sessions</p>
                </div>
            </div>
        </div>
    );
}

