from attr import define,field


# Docx文件段落数据
@define
class DocxParagraphData:
    content: str = field()


# Docx文件章节数据
@define
class DocxChapterData:
    # 章节编号
    sn: str = field(default='')
    # 章节标题
    title: str = field(default='')
    # 章节内容（不含子章节）
    paragraphs: list = field(factory=list)
    # 子章节
    subChapters: list = field(factory=list)

    def add_paragraph(self, paragraph: DocxParagraphData):
        self.paragraphs.append(paragraph)

    def add_sub_chapter(self, sub_chapter: 'DocxChapterData'):
        self.subChapters.append(sub_chapter)


@define
class DocxData:
    # 文件名
    fileName: str = field()
    # 文件路径
    filePath: str = field()
    # 章节
    chapters: list = field(factory=list)

    def add_chapter(self, chapter: DocxChapterData):
        self.chapters.append(chapter)