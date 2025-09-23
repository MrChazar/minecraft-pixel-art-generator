import { Injectable, NotFoundException } from '@nestjs/common';
import { CreateImageDto } from './dto/create-image.dto';
import { PrismaService } from '../prisma/prisma.service';
import { HttpService } from '@nestjs/axios';
import { firstValueFrom } from 'rxjs';
import { randomUUID } from 'node:crypto';
import { promises as fs } from 'node:fs';
import { join } from 'node:path';
import { CreateImageResponseDto } from './dto/create-image-response.dto';
import { UpdateImageDto } from './dto/update-image.dto';

@Injectable()
export class ImageService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly http: HttpService,
  ) {}

  async create(createImageDto: CreateImageDto, userEmail:string) : Promise<CreateImageResponseDto>  {
    const image = await firstValueFrom(this.http.post("http://localhost:8000/generate",{
    "is_block":createImageDto.is_block,
    "type_idx":createImageDto.type_idx,
    "color_idx":createImageDto.color_idx,
    "temperature":createImageDto.temperature
    }, {responseType:'arraybuffer'}))
    const filename = `${randomUUID()}.png` ;
    const filePath = join(process.cwd(), 'media', filename);

    await fs.writeFile(filePath,Buffer.from(image.data),'binary')

    return await this.prisma.image.create({data:{
      url:`/media/${filename}`,
      title: createImageDto.title ?? 'Default title',
      user_email:userEmail,
    }}) as CreateImageResponseDto
  }

  async findAll() {
    return await this.prisma.image.findMany() 
  }

  async findOne(url: string) {
    const image = await this.prisma.image.findUnique({ where: { url:`/media/${url}` } });
    if (image === null) {
      throw new NotFoundException("The image with given url doesn't exist");
    }
    return image;
  }

  async update(url: string, updateImageDto: UpdateImageDto) {
    const image = await this.prisma.image.findUnique({where:{url}})
    if (image ===null){
      throw new NotFoundException("This image does not exist")
    }
    if (updateImageDto.title != null){
      image.title = updateImageDto.title
    }
    return await this.prisma.image.update({where:{url},
      data:{...image}}
    )
  }

  async remove(url: string) {
    return await this.prisma.image.delete({where:{url}})
  }

  async findAllByUser(email: string) {
  return await this.prisma.image.findMany({
    where: { user_email:email },
    orderBy: { createdAt: 'desc' },
  });
}
}
