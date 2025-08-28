# localization.py

translations = {
    'en': {
        # Interface elements
        'app_title': "🏥 MedRAX",
        'app_subtitle': "Medical Reasoning Agent", # Original subtitle was more CXR specific
        'chat_placeholder': "Ask about the uploaded file or type your query...",
        'agent_label': "Agent",
        'image_label': "Image",
        'upload_xray_btn': "📎 Upload X-Ray",
        'upload_dicom_btn': "📄 Upload DICOM",
        'upload_blood_test_btn': "🩸 Upload Blood Test (.pdf, .csv)",
        'upload_mr_ct_btn': "🧲 Upload MR/CT Series (ZIP)",
        'upload_ophthalmic_btn': "👁️ Upload Eye FFA/OCT",
        'clear_chat_btn': "Clear Chat",
        'new_thread_btn': "New Thread",
        'uploaded_file_notification': "(System notification: User uploaded '{filename}')",

        # Tool result titles/common phrases
        'tool_output_title': "⚙️ Tool Output: {tool_name}",
        'image_from_tool_title': "🖼️ Image from tool: {tool_name}", # Kept from original, but might be replaced by specific titles
        'error_message_title': "Error",
        'visualized_image_title': "Visualized Image",
        'imaging_series_analysis_title': "Imaging Series Analysis",
        'imaging_series_results_title': "Imaging Series Analysis Results", # Used in metadata
        'representative_slice_title': "Representative Slice",
        'ophthalmic_analysis_title': "Ophthalmic Analysis",
        'ophthalmic_image_frame_title': "Ophthalmic Image/Frame",
        'extracted_video_frames_title': "Extracted Video Frames",

        # Phrases used in custom tool output formatting
        'results_title_prefix': "**{tool_name} Results:**",
        'status_label': "- Status: {status}",
        'slices_found_label': "- Slices Found: {count}",
        'images_in_series_label': "- Images in Series: {count}",
        'image_type_label': "- Image Type: {type}",
        'key_series_metadata_label': "- Key Series Metadata:",
        'representative_slice_info_label': "- Representative Slice Info:",
        'file_metadata_label': "- File Metadata:",
        'additional_frames_label': "- Additional Extracted Frames ({count}):",
        # Blood test specific (though not yet explicitly localized in interface.py formatting)
        'blood_test_interpretation_title': "Blood Test Interpretation",


    },
    'tr': {
        # Interface elements
        'app_title': "🏥 MedRAX", # Logo/brand name likely stays
        'app_subtitle': "Tıbbi Muhakeme Asistanı",
        'chat_placeholder': "Yüklenen dosya hakkında soru sorun veya sorgunuzu yazın...",
        'agent_label': "Asistan",
        'image_label': "Görüntü",
        'upload_xray_btn': "📎 Röntgen Yükle",
        'upload_dicom_btn': "📄 DICOM Yükle",
        'upload_blood_test_btn': "🩸 Kan Testi Yükle (.pdf, .csv)",
        'upload_mr_ct_btn': "🧲 MR/CT Serisi Yükle (ZIP)",
        'upload_ophthalmic_btn': "👁️ Göz FFA/OCT Yükle",
        'clear_chat_btn': "Sohbeti Temizle",
        'new_thread_btn': "Yeni Konu",
        'uploaded_file_notification': "(Sistem bildirimi: Kullanıcı '{filename}' dosyasını yükledi)",

        # Tool result titles/common phrases
        'tool_output_title': "⚙️ Araç Çıktısı: {tool_name}",
        'image_from_tool_title': "🖼️ Araçtan görüntü: {tool_name}",
        'error_message_title': "Hata",
        'visualized_image_title': "Görselleştirilmiş Görüntü",
        'imaging_series_analysis_title': "Görüntüleme Serisi Analizi",
        'imaging_series_results_title': "Görüntüleme Serisi Analiz Sonuçları",
        'representative_slice_title': "Temsili Kesit",
        'ophthalmic_analysis_title': "Oftalmik Analiz",
        'ophthalmic_image_frame_title': "Oftalmik Görüntü/Kare",
        'extracted_video_frames_title': "Çıkarılan Video Kareleri",

        # Phrases used in custom tool output formatting
        'results_title_prefix': "**{tool_name} Sonuçları:**",
        'status_label': "- Durum: {status}",
        'slices_found_label': "- Bulunan Kesit Sayısı: {count}",
        'images_in_series_label': "- Serideki Görüntü Sayısı: {count}",
        'image_type_label': "- Görüntü Türü: {type}",
        'key_series_metadata_label': "- Temel Seri Metaverileri:",
        'representative_slice_info_label': "- Temsili Kesit Bilgileri:",
        'file_metadata_label': "- Dosya Metaverileri:",
        'additional_frames_label': "- Ek Çıkarılan Kareler ({count}):",
        'blood_test_interpretation_title': "Kan Testi Yorumu",
    }
}

# Default language
DEFAULT_LANG = 'en'

def get_string(lang: str, key: str, **kwargs) -> str:
    """
    Retrieves a translated string for the given language and key.
    Falls back to the default language (English) if the key is not found in the specified language.
    If the key is not found in the default language either, it returns the key itself.
    Supports basic templating with **kwargs.
    """
    selected_lang_dict = translations.get(lang, translations[DEFAULT_LANG])
    val = selected_lang_dict.get(key)

    if val is None: # Fallback to default language if key not in selected lang
        val = translations[DEFAULT_LANG].get(key, key)

    if kwargs:
        try:
            return val.format(**kwargs)
        except KeyError: # In case a format key is missing in kwargs, return original string
            return val
    return val

def get_current_language_options() -> list:
    """Returns a list of available language codes and their names for UI display."""
    # This can be expanded with full language names if needed
    return [("English", "en"), ("Türkçe", "tr")]

if __name__ == '__main__':
    # Test the function
    print("--- English ---")
    print(get_string('en', 'app_title'))
    print(get_string('en', 'upload_xray_btn'))
    print(get_string('en', 'non_existent_key')) # Test fallback to key
    print(get_string('en', 'status_label', status='Completed'))


    print("\n--- Turkish ---")
    print(get_string('tr', 'app_title'))
    print(get_string('tr', 'upload_xray_btn'))
    print(get_string('tr', 'chat_placeholder'))
    print(get_string('tr', 'non_existent_key_in_tr_but_in_en', fallback_key_test="This should be key")) # Test fallback to EN then key
    print(get_string('tr', 'status_label', status='Tamamlandı'))


    print("\n--- Fallback from non-existent language ---")
    print(get_string('es', 'app_title')) # Should fallback to English
    print(get_string('es', 'upload_xray_btn'))
    print(get_string('es', 'status_label', status='Completado'))


    # Test a key only in EN, accessed via TR
    translations['en']['only_in_en'] = "English Only String"
    print(get_string('tr', 'only_in_en'))
    del translations['en']['only_in_en']

    print(get_current_language_options())
