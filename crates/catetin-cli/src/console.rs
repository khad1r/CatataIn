#[cfg(windows)]
extern "system" {
    fn GetStdHandle(nStdHandle: u32) -> isize;
    fn GetConsoleMode(hConsoleHandle: isize, lpMode: *mut u32) -> i32;
    fn SetConsoleMode(hConsoleHandle: isize, dwMode: u32) -> i32;
    fn FlushConsoleInputBuffer(hConsoleInput: isize) -> i32;
}

#[cfg(windows)]
const STD_INPUT_HANDLE: u32 = 0xFFFFFFF6;
#[cfg(windows)]
const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5;
#[cfg(windows)]
const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;

#[cfg(windows)]
const ENABLE_LINE_INPUT: u32 = 0x0002;
#[cfg(windows)]
const ENABLE_ECHO_INPUT: u32 = 0x0004;
#[cfg(windows)]
const ENABLE_MOUSE_INPUT: u32 = 0x0010;
#[cfg(windows)]
const ENABLE_QUICK_EDIT_MODE: u32 = 0x0040;
#[cfg(windows)]
const ENABLE_EXTENDED_FLAGS: u32 = 0x0080;

#[cfg(windows)]
pub struct RawConsoleGuard {
    in_handle: isize,
    out_handle: isize,
    orig_in_mode: Option<u32>,
    orig_out_mode: Option<u32>,
}

#[cfg(windows)]
impl RawConsoleGuard {
    pub fn enter() -> Self {
        unsafe {
            let in_handle = GetStdHandle(STD_INPUT_HANDLE);
            let out_handle = GetStdHandle(STD_OUTPUT_HANDLE);

            let mut orig_out_mode = None;
            let mut out_mode: u32 = 0;
            if GetConsoleMode(out_handle, &mut out_mode) != 0 {
                orig_out_mode = Some(out_mode);
                let _ = SetConsoleMode(out_handle, out_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
            }

            let mut orig_in_mode = None;
            let mut in_mode: u32 = 0;
            if GetConsoleMode(in_handle, &mut in_mode) != 0 {
                orig_in_mode = Some(in_mode);
                let raw_in_mode = (in_mode
                    & !(ENABLE_LINE_INPUT
                        | ENABLE_ECHO_INPUT
                        | ENABLE_MOUSE_INPUT))
                    | ENABLE_EXTENDED_FLAGS
                    | ENABLE_QUICK_EDIT_MODE;
                let _ = SetConsoleMode(in_handle, raw_in_mode);
                let _ = FlushConsoleInputBuffer(in_handle);
            }

            Self {
                in_handle,
                out_handle,
                orig_in_mode,
                orig_out_mode,
            }
        }
    }
}

#[cfg(windows)]
impl Drop for RawConsoleGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(in_mode) = self.orig_in_mode {
                let _ = SetConsoleMode(self.in_handle, in_mode);
            }
            if let Some(out_mode) = self.orig_out_mode {
                let _ = SetConsoleMode(self.out_handle, out_mode);
            }
        }
    }
}

#[cfg(not(windows))]
pub struct RawConsoleGuard;

#[cfg(not(windows))]
impl RawConsoleGuard {
    pub fn enter() -> Self {
        Self
    }
}
