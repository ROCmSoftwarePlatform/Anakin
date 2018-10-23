/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/db.hpp>
#include <miopen/db_record.hpp>
#include <miopen/errors.hpp>
#include <miopen/logger.hpp>
#include <miopen/md5.hpp>
#include <sys/stat.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <ios>
#include <mutex>
#include <array>
#include <sstream>
#include <iomanip>

#if (__cplusplus >= 201402L)
#include <shared_mutex>
#endif

#include <string>

#include <miopen/boost_utils.hpp>

namespace miopen {

std::timed_mutex mtx;

struct RecordPositions {
    std::streamoff begin = -1;
    std::streamoff end   = -1;
};

auto genMD5(std::string s) {
    std::array<unsigned char, MD5_DIGEST_LENGTH> result{};
    MD5(reinterpret_cast<const unsigned char*>(s.data()), s.length(), result.data());

    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (auto c : result)
        sout << std::setw(2) << int{c};

    return sout.str();
}

std::string LockFilePath(const std::string& filename_) {
    auto directory = temp_directory_path() + "/" + "miopen-lockfiles";
    MIOPEN_LOG_I("LockFilePath() , filename : " << filename_ << "  ,directory: " << directory);
    if (!exists(directory)) {
        create_directories(directory);
        permissions(directory, S_IRWXU | S_IRWXG | S_IRWXO);
    }

    const auto hash = genMD5(parent_path(filename_));
    const auto file = directory + "/" + (hash + "_" + filename(filename_) + ".lock");
    MIOPEN_LOG_I("LockFilePath() , file : " << file);
    return file;
}

Db::Db(const std::string& filename_, bool is_system) :
        filename(filename_), /*
         lock_file(LockFile::Get(LockFilePath(filename_).c_str())),*/
        warn_if_unreadable(is_system) {

    if (!is_system) {
        auto directory = remove_filename(filename_);
        if (!(exists(directory)) && !create_directories(directory))
            MIOPEN_LOG_W("Unable to create a directory: " << directory);
    }
}

#define MIOPEN_VALIDATE_LOCK(lock) \
    do {                           \
        if (!(lock))               \
            MIOPEN_LOG_E("FAIL");  \
    } while (false)

static std::chrono::seconds GetLockTimeout() {
    return std::chrono::seconds{60};
}
/*
using exclusive_lock = std::unique_lock<LockFile>;
#if (__cplusplus >= 201402L)
using shared_lock = std::shared_lock<LockFile>;
#else
using shared_lock = std::unique_lock<LockFile>;
#endif
*/
DbRecord Db::FindRecord(const std::string& key) {
    // const auto lock = shared_lock(lock_file, GetLockTimeout());
    // MIOPEN_VALIDATE_LOCK(lock);
    std::unique_lock<std::timed_mutex> lck(mtx, std::defer_lock);
    const auto lock = lck.try_lock_for(GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    return FindRecordUnsafe(key, nullptr);
}

bool Db::StoreRecord(const DbRecord& record) {
    // const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    // MIOPEN_VALIDATE_LOCK(lock);
    std::unique_lock<std::timed_mutex> lck(mtx, std::defer_lock);
    const auto lock = lck.try_lock_for(GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    return StoreRecordUnsafe(record);
}

bool Db::UpdateRecord(DbRecord& record) {
    // const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    // MIOPEN_VALIDATE_LOCK(lock);
    std::unique_lock<std::timed_mutex> lck(mtx, std::defer_lock);
    const auto lock = lck.try_lock_for(GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    return UpdateRecordUnsafe(record);
}

bool Db::RemoveRecord(const std::string& key) {
    // const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    // MIOPEN_VALIDATE_LOCK(lock);
    std::unique_lock<std::timed_mutex> lck(mtx, std::defer_lock);
    const auto lock = lck.try_lock_for(GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);
    return RemoveRecordUnsafe(key);
}

bool Db::Remove(const std::string& key, const std::string& id) {
    // const auto lock = exclusive_lock(lock_file, GetLockTimeout());
    // MIOPEN_VALIDATE_LOCK(lock);
    std::unique_lock<std::timed_mutex> lck(mtx, std::defer_lock);
    const auto lock = lck.try_lock_for(GetLockTimeout());
    MIOPEN_VALIDATE_LOCK(lock);

    auto record = FindRecordUnsafe(key, nullptr);

    if (!record.availableDb)
        return false;
    bool erased = record.EraseValues(id);
    if (!erased)
        return false;
    return StoreRecordUnsafe(record);
}

DbRecord Db::FindRecordUnsafe(const std::string& key, RecordPositions* pos) {
    if (pos != nullptr) {
        pos->begin = -1;
        pos->end   = -1;
    }

    MIOPEN_LOG_I("Looking for key: " << key);
    DbRecord empty_record(key);

    std::ifstream file(filename); // open db file. file name. example: "gfx900_64.cd.pad"

    if (!file) {
        if (warn_if_unreadable)
            MIOPEN_LOG_W("File is unreadable: " << filename);
        else
            MIOPEN_LOG_I("File is unreadable: " << filename);
        empty_record.availableDb = false;
        return empty_record;
    }

    int n_line = 0;
    while (true) {
        std::string line;
        const auto line_begin = file.tellg();
        if (!std::getline(file, line))
            break;
        ++n_line;
        const auto next_line_begin = file.tellg();

        const auto key_size = line.find('=');
        const bool is_key   = (key_size != std::string::npos && key_size != 0);
        if (!is_key) {
            if (!line.empty()) // Do not blame empty lines.
            {
                MIOPEN_LOG_E("Ill-formed record: key not found: " << filename << "#" << n_line);
            }
            continue;
        }
        const auto current_key = line.substr(0, key_size);

        if (current_key != key) {
            continue;
        }
        MIOPEN_LOG_I("Key match: " << current_key);
        const auto contents = line.substr(key_size + 1);

        if (contents.empty()) {
            MIOPEN_LOG_E(
                    "None contents under the key: " << current_key << " form file " << filename
                                                    << "#" << n_line);
            continue;
        }
        MIOPEN_LOG_I("Contents found: " << contents);

        DbRecord record(key);
        const bool is_parse_ok = record.ParseContents(contents);

        if (!is_parse_ok) {
            MIOPEN_LOG_E(
                    "Error parsing payload under the key: " << current_key << " form file "
                                                            << filename << "#" << n_line);
            MIOPEN_LOG_E("Contents: " << contents);
        }
        // A record with matching key have been found.
        if (pos != nullptr) {
            pos->begin = line_begin;
            pos->end   = next_line_begin;
        }
        record.availableDb = true;
        return record;
    }
    // Record was not found
    empty_record.availableDb = false;
    return empty_record;
}

static void Copy(std::istream& from, std::ostream& to, std::streamoff count) {
    constexpr auto buffer_size = 4 * 1024 * 1024;
    char buffer[buffer_size];
    auto left = count;

    while (left > 0 && !from.eof()) {
        const auto to_read = std::min<std::streamoff>(left, buffer_size);
        from.read(buffer, to_read);
        const auto read = from.gcount();
        to.write(buffer, read);
        left -= read;
    }
}

bool Db::FlushUnsafe(const DbRecord& record, const RecordPositions* pos) {
    assert(pos);

    if (pos->begin < 0 || pos->end < 0) {
        std::ofstream file(filename, std::ios::app);

        if (!file) {
            MIOPEN_LOG_E("File is unwritable: " << filename);
            return false;
        }

        (void)file.tellp();
        record.WriteContents(file);
    } else {
        std::ifstream from(filename, std::ios::ate);

        if (!from) {
            MIOPEN_LOG_E("File is unreadable: " << filename);
            return false;
        }

        const auto temp_name = filename + ".temp";
        std::ofstream to(temp_name);

        if (!to) {
            MIOPEN_LOG_E("Temp file is unwritable: " << temp_name);
            return false;
        }

        const auto from_size = from.tellg();
        from.seekg(std::ios::beg);

        Copy(from, to, pos->begin);
        record.WriteContents(to);
        from.seekg(pos->end);
        Copy(from, to, from_size - pos->end);

        from.close();
        to.close();

        std::remove(filename.c_str());
        std::rename(temp_name.c_str(), filename.c_str());
        /// \todo What if rename fails? Thou shalt not loose the original file.
    }
    return true;
}

bool Db::StoreRecordUnsafe(const DbRecord& record) {
    MIOPEN_LOG_I("Storing record: " << record.key);
    RecordPositions pos;
    const auto old_record = FindRecordUnsafe(record.key, &pos);
    if (!old_record.availableDb)
        return false;
    return FlushUnsafe(record, &pos);
}

bool Db::UpdateRecordUnsafe(DbRecord& record) {
    RecordPositions pos;
    const auto old_record = FindRecordUnsafe(record.key, &pos);
    DbRecord new_record(record);
    if (old_record.availableDb) { // ethan wu
        new_record.Merge(old_record);
        MIOPEN_LOG_I("Updating record: " << record.key);
    } else {
        MIOPEN_LOG_I("Storing record: " << record.key);
    }
    bool result = FlushUnsafe(new_record, &pos);
    if (result)
        record = std::move(new_record);
    return result;
}

bool Db::RemoveRecordUnsafe(const std::string& key) {
    // Create empty record with same key and replace original with that
    // This will remove record
    MIOPEN_LOG_I("Removing record: " << key);
    RecordPositions pos;
    FindRecordUnsafe(key, &pos);
    const DbRecord empty_record(key);
    return FlushUnsafe(empty_record, &pos);
}

} // namespace miopen
