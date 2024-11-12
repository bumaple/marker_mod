from typing import Tuple

from attr import define,field
from loguru import logger

# Docx文件段落数据
@define
class DocxParagraphData:
    content: str = field()

    def jsonformat(self):
        return {
            "段落内容": self.content
        }

    def __str__(self):
        return self.content


# Docx文件章节数据
@define
class DocxChapterData:
    # 章节序号
    seq: int = field(default=1)
    # 章节关键字
    key: str = field(default='')
    # 章节编号
    sn: str = field(default='')
    # 章节标题
    title: str = field(default='')
    # 章节级别
    level: int = field(default=1)
    # 章节内容（不含子章节）
    paragraphs: list = field(factory=list)
    # 子章节
    subChapters: dict[str, 'DocxChapterData'] = field(factory=dict[str, 'DocxChapterData'])

    def jsonformat(self):
        common_data = {
            "章节序号": self.seq,
            "章节编号": self.sn,
            "章节标题": self.title,
            "章节级别": self.level,
        }
        if len(self.paragraphs) > 0:
            common_data["段落内容"] = [paragraph.__str__() for paragraph in self.paragraphs]
        if len(self.subChapters) > 0:
            common_data["子章节"] = [subChapter.jsonformat() for subChapter in self.subChapters.values()]
        return common_data

    def __str__(self):
        if len(self.subChapters) > 0:
            return f"[章节信息]：key={self.key} Sn={self.sn} Title={self.title} Seq:{self.seq} Sub:{[subChapter.__str__() for subChapter in self.subChapters.values()]}"
        else:
            return f"[章节信息]：key={self.key} Sn={self.sn} Title={self.title} Seq:{self.seq}"

    def add_paragraph(self, paragraph: DocxParagraphData):
        self.paragraphs.append(paragraph)

    def add_sub_chapter(self, sub_chapter: 'DocxChapterData'):
        self.subChapters[sub_chapter.key] = sub_chapter

    def find_chapter_by_key(self, key: str, sn: str, title: str, level: int, seq: int, count: int=1) -> Tuple['DocxChapterData', int]:
        key_list = key.split(".")
        chapter = None
        while count < level:
            key_str = '.'.join(key_list[0:(1 + count)])
            count += 1
            if chapter is None:
                chapter = self.subChapters.get(key_str)
                if chapter is not None:
                    logger.debug(f"DocxChapterData 找到所属章节: {chapter}")
            else:
                sub_chapter, seq = chapter.find_chapter_by_key(key=key_str, sn=sn, title=title, level=level, seq=seq, count=count)
                logger.debug(f"DocxChapterData 找到所属章节: {sub_chapter}")
                if sub_chapter is not None:
                    chapter = sub_chapter

        if chapter is None:
            chapter = DocxChapterData(key=key, sn=sn, title=title, level=level, seq=seq)
            self.add_sub_chapter(chapter)
            seq += 1
            logger.debug(f"DocxChapterData 新增子章节: {key}")
            logger.debug(f"──{self}")
            logger.debug(f"   └─{chapter}")
            return chapter, seq
        elif count == level:
            return chapter, seq


@define
class DocxData:
    # 文件名
    fileName: str = field()
    # 章节
    chapters: dict[str, DocxChapterData] = field(factory=dict[str, DocxChapterData])

    def jsonformat(self):
        return {
            "文件名": self.fileName,
            "章节": [chapter.jsonformat() for chapter in self.chapters.values()]
        }

    def to_json(self):
        return self.jsonformat()

    def add_chapter(self, chapter: DocxChapterData):
        self.chapters[chapter.key] = chapter

    def find_chapter_by_key(self, key: str, sn: str, title: str, level: int, seq: int) -> Tuple[DocxChapterData, int]:
        # logger.debug(f"DocxData 查找章节: {key}")
        chapter = self.chapters.get(key)
        if chapter is not None:
            logger.debug(f"DocxData 找到所属章节: {chapter}")
            return chapter, seq
        logger.debug(f"DocxData 未找到所属章节: {key}")
        # 按照.分割搜索子章节
        key_list = key.split(".")
        if len(key_list) > 1:
            chapter, seq = self.find_chapter_by_key(key=key_list[0], sn=sn, title=title, level=level, seq=seq)
            if chapter:
                return chapter.find_chapter_by_key(key=key, sn=sn, title=title, level=level, seq=seq)

        # 生成一个新实例
        seq += 1
        chapter = DocxChapterData(key=key, sn=sn, title=title, level=level, seq=seq)
        self.add_chapter(chapter)
        logger.debug(f"DocxData 新增章节: {chapter}")
        return chapter, seq