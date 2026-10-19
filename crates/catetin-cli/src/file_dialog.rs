use std::path::PathBuf;

#[cfg(windows)]
pub fn pick_save_file(default_filename: &str, title: &str) -> Option<PathBuf> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};

    #[repr(C)]
    #[allow(non_snake_case)]
    struct OPENFILENAMEW {
        lStructSize: u32,
        hwndOwner: usize,
        hInstance: usize,
        lpstrFilter: *const u16,
        lpstrCustomFilter: *mut u16,
        nMaxCustFilter: u32,
        nFilterIndex: u32,
        lpstrFile: *mut u16,
        nMaxFile: u32,
        lpstrFileTitle: *mut u16,
        nMaxFileTitle: u32,
        lpstrInitialDir: *const u16,
        lpstrTitle: *const u16,
        Flags: u32,
        nFileOffset: u16,
        nFileExtension: u16,
        lpstrDefExt: *const u16,
        lCustData: usize,
        lpfnHook: usize,
        lpTemplateName: *const u16,
        pvReserved: *mut std::ffi::c_void,
        dwReserved: u32,
        FlagsEx: u32,
    }

    #[link(name = "comdlg32")]
    extern "system" {
        fn GetSaveFileNameW(lpofn: *mut OPENFILENAMEW) -> i32;
    }

    const OFN_OVERWRITEPROMPT: u32 = 0x00000002;
    const OFN_PATHMUSTEXIST: u32 = 0x00000800;

    let mut buffer = [0u16; 1024];
    let default_wide: Vec<u16> = std::ffi::OsStr::new(default_filename)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let len = default_wide.len().min(buffer.len() - 1);
    buffer[..len].copy_from_slice(&default_wide[..len]);

    let filter_wide: Vec<u16> = "Markdown (*.md)\0*.md\0All Files (*.*)\0*.*\0\0"
        .encode_utf16()
        .collect();
    let title_wide: Vec<u16> = std::ffi::OsStr::new(title)
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let def_ext_wide: Vec<u16> = "md\0".encode_utf16().collect();

    let mut ofn: OPENFILENAMEW = unsafe { std::mem::zeroed() };
    ofn.lStructSize = std::mem::size_of::<OPENFILENAMEW>() as u32;
    ofn.lpstrFilter = filter_wide.as_ptr();
    ofn.nFilterIndex = 1;
    ofn.lpstrFile = buffer.as_mut_ptr();
    ofn.nMaxFile = buffer.len() as u32;
    ofn.lpstrTitle = title_wide.as_ptr();
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST;
    ofn.lpstrDefExt = def_ext_wide.as_ptr();

    let success = unsafe { GetSaveFileNameW(&mut ofn) };
    if success != 0 {
        let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
        let os_str = std::ffi::OsString::from_wide(&buffer[..end]);
        Some(PathBuf::from(os_str))
    } else {
        None
    }
}

#[cfg(all(not(windows), target_os = "linux"))]
pub fn pick_save_file(default_filename: &str, title: &str) -> Option<PathBuf> {
    let output = std::process::Command::new("zenity")
        .args([
            "--file-selection",
            "--save",
            "--confirm-overwrite",
            "--title",
            title,
            "--filename",
            default_filename,
        ])
        .output()
        .ok()?;

    if output.status.success() {
        let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            return Some(PathBuf::from(path_str));
        }
    }
    None
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn pick_save_file(_default_filename: &str, _title: &str) -> Option<PathBuf> {
    None
}
