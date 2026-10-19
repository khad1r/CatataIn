use std::io::{self, Write};

#[cfg(windows)]
extern "C" {
    fn _kbhit() -> i32;
    fn _getch() -> i32;
}

#[cfg(windows)]
extern "system" {
    fn GetStdHandle(nStdHandle: u32) -> isize;
    fn GetNumberOfConsoleInputEvents(hConsoleInput: isize, lpcNumberOfEvents: *mut u32) -> i32;
    fn ReadConsoleInputW(
        hConsoleInput: isize,
        lpBuffer: *mut InputRecord,
        nLength: u32,
        lpNumberOfEventsRead: *mut u32,
    ) -> i32;
}

#[cfg(windows)]
const STD_INPUT_HANDLE: u32 = 0xFFFFFFF6;
#[cfg(windows)]
const KEY_EVENT: u16 = 0x0001;
#[cfg(windows)]
const VK_BACK: u16 = 0x08;
#[cfg(windows)]
const VK_RETURN: u16 = 0x0D;
#[cfg(windows)]
const VK_ESCAPE: u16 = 0x1B;
#[cfg(windows)]
const VK_SPACE: u16 = 0x20;
#[cfg(windows)]
const VK_LEFT: u16 = 0x25;
#[cfg(windows)]
const VK_UP: u16 = 0x26;
#[cfg(windows)]
const VK_RIGHT: u16 = 0x27;
#[cfg(windows)]
const VK_DOWN: u16 = 0x28;
#[cfg(windows)]
const VK_PRIOR: u16 = 0x21;
#[cfg(windows)]
const VK_NEXT: u16 = 0x22;
#[cfg(windows)]
const VK_END: u16 = 0x23;
#[cfg(windows)]
const VK_HOME: u16 = 0x24;

#[cfg(windows)]
const MOUSE_EVENT: u16 = 0x0002;
#[cfg(windows)]
const MOUSE_WHEELED: u32 = 0x0004;

#[cfg(windows)]
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct KeyEventRecord {
    b_key_down: i32,
    w_repeat_count: u16,
    w_virtual_key_code: u16,
    w_virtual_scan_code: u16,
    u_char: u16,
    dw_control_key_state: u32,
}

#[cfg(windows)]
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct MouseEventRecord {
    dw_mouse_position_x: i16,
    dw_mouse_position_y: i16,
    dw_button_state: u32,
    dw_control_key_state: u32,
    dw_event_flags: u32,
}

#[cfg(windows)]
#[repr(C)]
#[derive(Copy, Clone)]
union EventUnion {
    key_event: KeyEventRecord,
    mouse_event: MouseEventRecord,
}

#[cfg(windows)]
impl Default for EventUnion {
    fn default() -> Self {
        Self {
            key_event: KeyEventRecord::default(),
        }
    }
}

#[cfg(windows)]
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct InputRecord {
    event_type: u16,
    _pad: u16,
    event: EventUnion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Key {
    Up,
    Down,
    Left,
    Right,
    Enter,
    Escape,
    Backspace,
    Char(char),
    Other,
}

pub fn poll_key() -> Option<i32> {
    #[cfg(windows)]
    unsafe {
        let in_handle = GetStdHandle(STD_INPUT_HANDLE);
        let mut num_events: u32 = 0;
        if GetNumberOfConsoleInputEvents(in_handle, &mut num_events) != 0 {
            while num_events > 0 {
                let mut rec = InputRecord::default();
                let mut read: u32 = 0;
                if ReadConsoleInputW(in_handle, &mut rec, 1, &mut read) == 0 || read == 0 {
                    break;
                }
                num_events = num_events.saturating_sub(1);
                if rec.event_type == KEY_EVENT && rec.event.key_event.b_key_down != 0 {
                    let key_ev = rec.event.key_event;
                    match key_ev.w_virtual_key_code {
                        VK_UP => return Some(1001),
                        VK_DOWN => return Some(1002),
                        VK_PRIOR => return Some(1003),
                        VK_NEXT => return Some(1004),
                        VK_HOME => return Some(1005),
                        VK_END => return Some(1006),
                        VK_ESCAPE => return Some(27),
                        VK_RETURN => return Some(13),
                        VK_SPACE => return Some(b' ' as i32),
                        VK_BACK => return Some(8),
                        _ => {}
                    }

                    let ch = key_ev.u_char;
                    if ch == 27 {
                        let mut follow_events: u32 = 0;
                        if GetNumberOfConsoleInputEvents(in_handle, &mut follow_events) != 0 && follow_events > 0 {
                            let mut next_rec = InputRecord::default();
                            let mut next_read: u32 = 0;
                            if ReadConsoleInputW(in_handle, &mut next_rec, 1, &mut next_read) != 0 && next_read > 0 {
                                if next_rec.event.key_event.u_char == b'[' as u16 {
                                    let mut code_rec = InputRecord::default();
                                    let mut code_read: u32 = 0;
                                    if ReadConsoleInputW(in_handle, &mut code_rec, 1, &mut code_read) != 0 && code_read > 0 {
                                        match code_rec.event.key_event.u_char as u8 {
                                            b'A' => return Some(1001), // Up
                                            b'B' => return Some(1002), // Down
                                            b'5' => return Some(1003), // Page Up
                                            b'6' => return Some(1004), // Page Down
                                            b'H' => return Some(1005), // Home
                                            b'F' => return Some(1006), // End
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            return None;
                        }
                        return Some(27);
                    } else if ch != 0 {
                        return Some(ch as i32);
                    }
                } else if rec.event_type == MOUSE_EVENT {
                    let mouse_ev = rec.event.mouse_event;
                    if (mouse_ev.dw_event_flags & MOUSE_WHEELED) != 0 {
                        let delta = (mouse_ev.dw_button_state as i32) >> 16;
                        if delta > 0 {
                            return Some(1001); // Mouse wheel up -> Scroll Up
                        } else if delta < 0 {
                            return Some(1002); // Mouse wheel down -> Scroll Down
                        }
                    }
                }
            }
            None
        } else if _kbhit() != 0 {
            let ch = _getch();
            if ch == 0 || ch == 0xE0 {
                if _kbhit() != 0 {
                    let ext = _getch();
                    match ext {
                        72 => Some(1001), // Up
                        80 => Some(1002), // Down
                        73 => Some(1003), // Page Up
                        81 => Some(1004), // Page Down
                        71 => Some(1005), // Home
                        79 => Some(1006), // End
                        _ => None,
                    }
                } else {
                    None
                }
            } else {
                Some(ch)
            }
        } else {
            None
        }
    }
    #[cfg(not(windows))]
    {
        None
    }
}

pub fn read_key_blocking() -> Key {
    #[cfg(windows)]
    unsafe {
        let in_handle = GetStdHandle(STD_INPUT_HANDLE);
        let mut num_events: u32 = 0;
        if GetNumberOfConsoleInputEvents(in_handle, &mut num_events) != 0 {
            loop {
                let mut rec = InputRecord::default();
                let mut read: u32 = 0;
                if ReadConsoleInputW(in_handle, &mut rec, 1, &mut read) == 0 || read == 0 {
                    return Key::Other;
                }
                if rec.event_type == KEY_EVENT && rec.event.key_event.b_key_down != 0 {
                    let key_ev = rec.event.key_event;
                    match key_ev.w_virtual_key_code {
                        VK_UP => return Key::Up,
                        VK_DOWN => return Key::Down,
                        VK_LEFT => return Key::Left,
                        VK_RIGHT => return Key::Right,
                        VK_RETURN => return Key::Enter,
                        VK_ESCAPE => return Key::Escape,
                        VK_BACK => return Key::Backspace,
                        VK_SPACE => return Key::Char(' '),
                        _ => {}
                    }
                    let ch = key_ev.u_char;
                    if ch == 27 {
                        let mut follow_events: u32 = 0;
                        for _ in 0..6 {
                            if GetNumberOfConsoleInputEvents(in_handle, &mut follow_events) != 0 && follow_events > 0 {
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_millis(3));
                        }
                        if follow_events > 0 {
                            let mut next_rec = InputRecord::default();
                            let mut next_read: u32 = 0;
                            if ReadConsoleInputW(in_handle, &mut next_rec, 1, &mut next_read) != 0 && next_read > 0 {
                                if next_rec.event.key_event.u_char as u8 == b'[' {
                                    let mut code_rec = InputRecord::default();
                                    let mut code_read: u32 = 0;
                                    if ReadConsoleInputW(in_handle, &mut code_rec, 1, &mut code_read) != 0 && code_read > 0 {
                                        match code_rec.event.key_event.u_char as u8 {
                                            b'A' => return Key::Up,
                                            b'B' => return Key::Down,
                                            b'C' => return Key::Right,
                                            b'D' => return Key::Left,
                                            _ => {}
                                        }
                                    }
                                }
                            }
                            continue;
                        }
                        return Key::Escape;
                    } else if ch != 0 {
                        match ch as u8 {
                            13 | 10 => return Key::Enter,
                            8 | 127 => return Key::Backspace,
                            b'k' | b'K' => return Key::Up,
                            b'j' | b'J' => return Key::Down,
                            b'h' | b'H' => return Key::Left,
                            b'l' | b'L' => return Key::Right,
                            b' ' => return Key::Char(' '),
                            b @ 32..=126 => return Key::Char(b as char),
                            _ => {}
                        }
                    }
                }
            }
        }

        let ch = _getch();
        if ch < 0 {
            Key::Other
        } else if ch == 0 || ch == 0xE0 {
            let ext = _getch();
            match ext {
                72 => Key::Up,
                80 => Key::Down,
                75 => Key::Left,
                77 => Key::Right,
                _ => Key::Other,
            }
        } else if ch == 27 {
            if _kbhit() != 0 {
                let next = _getch();
                if next == b'[' as i32 && _kbhit() != 0 {
                    let code = _getch();
                    match code as u8 {
                        b'A' => Key::Up,
                        b'B' => Key::Down,
                        b'C' => Key::Right,
                        b'D' => Key::Left,
                        _ => Key::Escape,
                    }
                } else {
                    Key::Escape
                }
            } else {
                Key::Escape
            }
        } else {
            let b = ch as u8;
            match b {
                13 | 10 => Key::Enter,
                8 | 127 => Key::Backspace,
                b'k' | b'K' => Key::Up,
                b'j' | b'J' => Key::Down,
                b'h' | b'H' => Key::Left,
                b'l' | b'L' => Key::Right,
                b' ' => Key::Char(' '),
                32..=126 => Key::Char(b as char),
                _ => Key::Other,
            }
        }
    }
    #[cfg(not(windows))]
    {
        use std::io::Read;
        let mut buf = [0u8; 4];
        let mut stdin = std::io::stdin();
        if let Ok(n) = stdin.read(&mut buf) {
            if n >= 3 && buf[0] == 27 && buf[1] == b'[' {
                match buf[2] {
                    b'A' => Key::Up,
                    b'B' => Key::Down,
                    b'C' => Key::Right,
                    b'D' => Key::Left,
                    _ => Key::Other,
                }
            } else if n >= 1 {
                match buf[0] {
                    13 | 10 => Key::Enter,
                    27 => Key::Escape,
                    8 | 127 => Key::Backspace,
                    b'k' | b'K' => Key::Up,
                    b'j' | b'J' => Key::Down,
                    b'h' | b'H' => Key::Left,
                    b'l' | b'L' => Key::Right,
                    c if (32..=126).contains(&c) => Key::Char(c as char),
                    _ => Key::Other,
                }
            } else {
                Key::Other
            }
        } else {
            Key::Other
        }
    }
}

pub fn read_single_choice(prompt: &str, valid_keys: &[char], default: char) -> char {
    print!("{}", prompt);
    let _ = io::stdout().flush();

    loop {
        let key = read_key_blocking();
        match key {
            Key::Enter => {
                println!("{}", default);
                let _ = io::stdout().flush();
                return default;
            }
            Key::Char(c) => {
                let lower = c.to_ascii_lowercase();
                if valid_keys.iter().any(|&v| v.to_ascii_lowercase() == lower) {
                    println!("{}", c);
                    let _ = io::stdout().flush();
                    return c;
                }
            }
            Key::Escape => {
                println!("{}", default);
                let _ = io::stdout().flush();
                return default;
            }
            _ => {}
        }
    }
}

pub fn read_confirm(prompt: &str, default_yes: bool) -> bool {
    print!("{}", prompt);
    let _ = io::stdout().flush();

    loop {
        let key = read_key_blocking();
        match key {
            Key::Enter => {
                let s = if default_yes { "Y" } else { "N" };
                println!("{}", s);
                let _ = io::stdout().flush();
                return default_yes;
            }
            Key::Char('y') | Key::Char('Y') => {
                println!("Y");
                let _ = io::stdout().flush();
                return true;
            }
            Key::Char('n') | Key::Char('N') => {
                println!("N");
                let _ = io::stdout().flush();
                return false;
            }
            Key::Escape => {
                println!("N");
                let _ = io::stdout().flush();
                return false;
            }
            _ => {}
        }
    }
}

#[allow(dead_code)]
pub fn read_line(prompt: &str) -> String {
    print!("{}", prompt);
    let _ = io::stdout().flush();

    #[cfg(windows)]
    {
        let mut buf = String::new();
        loop {
            let key = read_key_blocking();
            match key {
                Key::Enter => {
                    println!();
                    let _ = io::stdout().flush();
                    return buf.trim().to_string();
                }
                Key::Escape => {
                    println!();
                    let _ = io::stdout().flush();
                    return String::new();
                }
                Key::Backspace => {
                    if !buf.is_empty() {
                        buf.pop();
                        print!("\x08 \x08");
                        let _ = io::stdout().flush();
                    }
                }
                Key::Char(c) => {
                    buf.push(c);
                    print!("{}", c);
                    let _ = io::stdout().flush();
                }
                _ => {}
            }
        }
    }
    #[cfg(not(windows))]
    {
        let mut line = String::new();
        let _ = io::stdin().read_line(&mut line);
        line.trim().to_string()
    }
}
