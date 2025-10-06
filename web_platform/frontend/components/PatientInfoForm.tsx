interface PatientInfo {
    name: string;
    age: string;
    gender: string;
    notes: string;
}

interface PatientInfoFormProps {
    patientInfo: PatientInfo;
    onChange: (info: PatientInfo) => void;
    onClose: () => void;
}

export default function PatientInfoForm({ patientInfo, onChange, onClose }: PatientInfoFormProps) {
    return (
        <div className="bg-zinc-900 border-b border-zinc-800 p-4">
            <div className="max-w-2xl">
                <h3 className="text-sm font-semibold text-zinc-300 mb-3">Patient Information</h3>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-xs text-zinc-500 mb-1">Name</label>
                        <input
                            type="text"
                            value={patientInfo.name}
                            onChange={(e) => onChange({ ...patientInfo, name: e.target.value })}
                            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-sm"
                            placeholder="Patient name"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-zinc-500 mb-1">Age</label>
                        <input
                            type="text"
                            value={patientInfo.age}
                            onChange={(e) => onChange({ ...patientInfo, age: e.target.value })}
                            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-sm"
                            placeholder="Age"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-zinc-500 mb-1">Gender</label>
                        <select
                            value={patientInfo.gender}
                            onChange={(e) => onChange({ ...patientInfo, gender: e.target.value })}
                            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-sm"
                        >
                            <option value="">Select</option>
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                    <div className="col-span-2">
                        <label className="block text-xs text-zinc-500 mb-1">Clinical Notes</label>
                        <textarea
                            value={patientInfo.notes}
                            onChange={(e) => onChange({ ...patientInfo, notes: e.target.value })}
                            className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded text-sm"
                            rows={3}
                            placeholder="Clinical notes..."
                        />
                    </div>
                </div>
                <div className="flex gap-2 mt-4">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                    >
                        Save
                    </button>
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-zinc-800 hover:bg-zinc-700 rounded text-sm"
                    >
                        Cancel
                    </button>
                </div>
            </div>
        </div>
    );
}

